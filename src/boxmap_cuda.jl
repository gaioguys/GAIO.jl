#export i32, ui32

struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

struct BoxMapGPUCache{SZ} end

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:gpu}) where {N,T}
    points_vec = cu(points)
    maxsize = CUDA.available_memory() ÷ (N * sizeof(Int32) * 2)
    return PointDiscretizedMap(map, domain, points_vec, BoxMapGPUCache{maxsize}())
end

for (key, val) in Dict(
            :map_boxes => :(!isnothing(hit) ? hit : 0i32), 
            :TransferOperator => :(!isnothing(hit) ? (key,hit) : (0i32,0i32))
        )

    kernel = Symbol(key, :_kernel!)
    @eval @muladd function $kernel(G, keys, points, out_keys, P, np, nk)
        ind = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x - 1i32
        stride = gridDim().x * blockDim().x
        len = nk * np - 1i32
        for i in ind : stride : len
            m, n = divrem(i, np) .+ 1i32
            key  = keys[n]
            box  = key_to_box(P, key)
            c, r = box.center, box.radius
            p    = points[m]
            fp   = G(@. p * r + c)
            hit  = point_to_key(P, fp)
            out_keys[i+1] = $val
        end
    end
end

function map_boxes(g::SampledBoxMap{<:BoxMapGPUCache{SZ}}, source::BoxSet) where SZ
    P, keys = source.partition, Stateful(source.set)
    points = g.domain_points(P.domain.center, P.domain.radius)
    image = BoxSet(P, Set{Int32}())
    while !isnothing(keys.nextvalstate)
        stride = min(SZ, length(keys))
        in_keys = CuArray{Int32,1}(collect(take(keys, stride)))
        nk, np = Int32(length(in_keys)), Int32(length(points))
        out_keys = CuArray{Int32,1}(undef, nk * np)
        launch_kernel_then_sync!(nk * np, map_boxes_kernel!, g.map, in_keys, points, out_keys, P, nk, np)
        out_cpu = Array{Int32,1}(out_keys)
        y = Set(out_cpu)
        delete!(y, 0i32)
        union!(image, BoxSet(P, y))
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    return image
end

function TransferOperator(g::SampledBoxMap{<:BoxMapGPUCache{SZ}}, source::BoxSet{<:BoxPartition}) where SZ
    P, SZ2 = source.partition, SZ ÷ 2
    points = g.domain_points(P.domain.center, P.domain.radius)
    edges = Dict{Tuple{Int,Int},Float64}()
    boxlist = BoxList(P, collect(Int32, source.set))
    key_to_index = invert_vector(boxlist.keylist)
    key_to_index[0i32] = 0
    for i in 0:SZ2:max(0, length(boxlist)-SZ2)
        stride = min(SZ2, length(boxlist)-i)
        in_keys = CuArray{Int32,1}(boxlist.keylist[i+1:i+stride])
        nk, np = Int32(length(in_keys)), Int32(length(points))
        out_keys = CuArray{Tuple{Int32,Int32},1}(undef, nk * np)
        launch_kernel_then_sync!(nk * np, TransferOperator_kernel!, g.map, in_keys, points, out_keys, P, nk, np)
        out_cpu = Array{Tuple{Int32,Int32},1}(out_keys)
        inv_n = 1. / np
        for (key, hit) in out_cpu
            if hit in source.set
                e = (key_to_index[key], key_to_index[hit])
                edges[e] = get(edges, e, 0.) + inv_n
            end
        end
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(edges, (0,0))
    return TransferOperator(boxlist, edges)
end

# helper + compatibility functions
function launch_kernel_then_sync!(n, kernel, args...)
    compiled_kernel! = @cuda launch=false kernel(args...)
    config  = launch_configuration(compiled_kernel!.fun)
    threads = min(n, config.threads)
    blocks  = cld(n, threads)
    compiled_kernel!(args...; threads, blocks)
    CUDA.synchronize()
    return
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
    ) where {N,Float64,V<:AbstractArray{<:SVNT{N,Float64}}}

    CuArray{SVector{N,Float32},1}(x)
end