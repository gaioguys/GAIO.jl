#export i32, ui32

struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

struct BoxMapGPUCache end

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:gpu}) where {N,T}
    points_vec = cu(points)
    return PointDiscretizedMap(map, domain, points_vec, BoxMapGPUCache())
end

function map_boxes_kernel!(g, P, domain_points, in_keys, out_keys)
    nk, np = Int32.((length(in_keys), length(domain_points)))
    ind = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x #- 1i32
    stride = gridDim().x * blockDim().x
    len = nk * np #- 1i32
    for i in ind : stride : len
        m, n = Int32.(CartesianIndices((np, nk))[i].I)
        p    = domain_points[m]
        key  = in_keys[n]
        box  = key_to_box(P, key)
        c, r = box.center, box.radius
        fp   = g(@muladd p .* r .+ c)
        hit  = point_to_key(P, fp)
        out_keys[i] = isnothing(hit) ? cu_zero(P) : hit
    end
end

function map_boxes(
        g::SampledBoxMap{C,N,T,F,D,typeof(center)}, source::BoxSet{B,Q,S}
    ) where {C<:BoxMapGPUCache,N,T,F,D,B,Q<:BoxPartition,S}

    P, keys = source.partition, Stateful(source.set)
    p = ensure_implemented(g, P)
    np = length(p)
    K = cu_keytype(P)
    image = S()
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_keys = CuArray{K,1}(collect(take(keys, stride)))
        nk = length(in_keys)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        union!(image, out_cpu)
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(image, cu_zero(P))
    return BoxSet(P, image)
end

function construct_transfers(
        g::SampledBoxMap{C,M,V,F,L,typeof(center)}, source::BoxSet{R,Q,S}
    ) where {C<:BoxMapGPUCache,M,V,N,T,F,L,R<:Box{N,T},Q<:BoxPartition,S<:OrderedSet}

    P, keys = source.partition, Stateful(source.set)
    p = ensure_implemented(g, P)
    np = length(p)
    inv_n = 1 / np
    K = cu_keytype(P)
    D = Dict{Tuple{K,K},Float32}
    mat = D()
    variant_keys = S()
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_cpu = Array{K,1}(collect(take(keys, stride)))
        in_keys = CuArray{K,1}(in_cpu)
        nk = length(in_keys)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        Cart = CartesianIndices((np, nk))
        for i in 1:nk*np
            _, n = Cart[i].I
            key, hit = in_cpu[n], out_cpu[i]
            mat = mat ⊔ ((hit,key) => inv_n)
        end
        union!(variant_keys, setdiff(out_cpu, source.set))
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(variant_keys, cu_zero(P))
    return mat, variant_keys
end

# helper + compatibility functions
function launch_kernel!(n, kernel, args...)
    compiled_kernel! = @cuda launch=false kernel(args...)
    config  = launch_configuration(compiled_kernel!.fun)
    threads = min(n, config.threads)
    blocks  = cld(n, threads)
    compiled_kernel!(args...; threads, blocks)
    return
end

function launch_kernel_then_sync!(n, kernel, args...)
    launch_kernel!(n, kernel, args...)
    CUDA.synchronize()
    return
end

function available_array_memory()
    m = CUDA.MemoryInfo()
    return m.free_bytes + m.pool_reserved_bytes
end

cu_reduce(::Type{I}) where {I<:Integer} = Int32
cu_reduce(::Type{Int16}) = Int16
cu_reduce(::Type{Int8}) = Int8

cu_keytype(::BoxPartition{N,T,I,<:IndexLinear}) where {N,T,I} = cu_reduce(I)
cu_keytype(::BoxPartition{N,T,I,<:IndexCartesian}) where {N,T,I} = NTuple{N,cu_reduce(I)}

cu_zero(::BoxPartition{N,T,I,<:IndexLinear}) where {N,T,I} = zero(cu_reduce(I))
cu_zero(::BoxPartition{N,T,I,<:IndexCartesian}) where {N,T,I} = ntuple(_->zero(cu_reduce(I)), Val(N))

for adaptor in (CUDA.CuArrayAdaptor, CUDA.Adaptor), T in (:Float64, :J), I in (:Int64, :Int128, :J)
    @eval function Adapt.adapt_structure(
            a::$adaptor, b::BoxPartition{N,$T,$I,A}
        ) where {N,J,A}

        Adapt.adapt_storage(a,
            BoxPartition{N,Float32,Int32,A}(
                Box{N,Float32}(b.domain.center, b.domain.radius),
                SVector{N,Float32}(b.left), SVector{N,Float32}(b.scale),
                SVector{N,Int32}(b.dims), SVector{N,Int32}(b.dimsprod),
                b.indextype
            )
        )
    end
end

function Adapt.adapt_structure(
        ::CUDA.CuArrayAdaptor, x::V
    ) where {N,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F}}}

    CuArray{SVector{N,Float32},1}(x)
end

function ensure_implemented(g, P)
    points = g.domain_points(P.domain.center, P.domain.radius)

    if points isa AbstractArray
        p = cu(points)
    elseif points isa Base.Generator{<:AbstractArray}
        p = cu(points.iter)
    else
        @error "Test point sampling techniques other than Monte-Carlo not implemented"
    end

    return p
end
