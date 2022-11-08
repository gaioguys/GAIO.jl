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
        out_keys[i] = isnothing(hit) ? 0i32 : hit
    end
end

function TransferOperator_kernel!(g, P, domain_points, in_keys, out_keys)
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
        out_keys[i] = isnothing(hit) ? (0i32, 0i32) : (key, hit)
    end
end

function map_boxes(
        g::SampledBoxMap{C,N,T,F,D,typeof(center)}, source::BoxSet{B,Q,S}
    ) where {C<:BoxMapGPUCache,N,T,F,D,B,Q<:BoxPartition{<:Any,<:Any,<:Any,<:IndexLinear},S}

    P, keys = source.partition, Stateful(source.set)
    p = ensure_implemented(g, P)
    np = length(p)
    image = S()
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(Int32) * 10 * (N + 1) * np)
        )
        in_keys = CuArray{Int32,1}(collect(take(keys, stride)))
        nk = length(in_keys)
        out_keys = CuArray{Int32,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{Int32,1}(out_keys)
        union!(image, out_cpu)
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(image, 0i32)
    return BoxSet(P, image)
end

function TransferOperator(
        g::SampledBoxMap{C,N,T,F,D,typeof(center)}, source::BoxSet{B,Q,S}
    ) where {C<:BoxMapGPUCache,N,T,F,D,B<:BoxPartition{<:Any,<:Any,<:Any,<:IndexLinear},Q,S}

    P = source.partition
    points = ensure_implemented(g, P)
    np = length(points)
    inv_n = 1. / np
    edges = Dict{Tuple{Int,Int},Float64}()
    boxlist = BoxList(P, collect(Int32, source.set))
    key_to_index = invert_vector(boxlist.keylist)
    key_to_index[0i32] = 0
    SZ2 = min(
        length(boxlist),
        available_array_memory() ÷ (sizeof(Int32) * 10 * (N + 1) * np)
    ) ÷ 2
    for i in 0:SZ2:max(0, length(boxlist)-SZ2)
        in_keys = CuArray{Int32,1}(boxlist.keylist[i+1:i+SZ2])
        nk = Int32(length(in_keys))
        out_keys = CuArray{Tuple{Int32,Int32},1}(undef, nk * np)
        launch_kernel_then_sync!(nk * np, TransferOperator_kernel!, g.map, P, points, in_keys, out_keys)
        out_cpu = Array{Tuple{Int32,Int32},1}(out_keys)
        for (key, hit) in out_cpu
            if hit in source.set
                e = (key_to_index[key], key_to_index[hit])
                edges[e] = get(edges, e, 0.) + inv_n
            end
        end
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(edges, (0.,0.))
    return TransferOperator(boxlist, edges)
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

for adaptor in (CUDA.CuArrayAdaptor, CUDA.Adaptor), T in (:Float64, :J), I in (:Int64, :Int128, :J)
    @eval function Adapt.adapt_structure(
            a::$adaptor, b::BoxPartition{N,$T,$I}
        ) where {N,J}

        Adapt.adapt_storage(a,
            BoxPartition{N,Float32,Int32}(
                Box{N,Float32}(b.domain.center, b.domain.radius),
                SVector{N,Float32}(b.left), SVector{N,Float32}(b.scale),
                SVector{N,Int32}(b.dims), SVector{N,Int32}(b.dimsprod)
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
