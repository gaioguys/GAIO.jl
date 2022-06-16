struct BoxMapGPUCache{SZ}
    maxsize::Val{SZ}
end

for T in (:Float64, :J), I in (:Int64, :Int128, :J)
    @eval function Adapt.adapt_structure(a::CUDA.Adaptor, b::BoxPartition{N,$T,$I,D}) where {N,J,D}
        Adapt.adapt_storage(a,
            BoxPartition{N,Float32,Int32,D}(
                Box{N,Float32}(b.domain.center, b.domain.radius),
                SVector{N,Float32}(b.left), SVector{N,Float32}(b.scale),
                SVector{N,Int32}(b.dims), SVector{N,Int32}(b.dimsprod)
            )
        )
    end
end

for T in (:SVector, :NTuple)
    @eval function Adapt.adapt_structure(a::CUDA.CuArrayAdaptor, x::V) where {N,Float64,V<:AV{$T{N,Float64}}}
        x = map($T{N,Float32}, x)
        return Adapt.adapt_storage(a, x)
    end
end

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:gpu}) where {N,T}
    points_vec = cu(points)
    maxsize = min(prod(CUDA.max_grid_size), CUDA.totalmem(CUDA.device()))
    maxsize ÷= N * sizeof(Int32) * length(points_vec) * 2
    return PointDiscretizedMap(map, domain, points_vec, BoxMapGPUCache(Val(maxsize)))
end

@muladd function map_boxes_kernel!(G, keys, points, out_keys, P, np, nk)
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
        out_keys[i+1] = !isnothing(hit) ? hit : 0i32
    end
end

function map_boxes(g::SampledBoxMap{<:BoxMapGPUCache{SZ}}, source::BoxSet) where SZ
    P, keys = source.partition, Stateful(source.set)
    points = g.domain_points(P.domain.center, P.domain.radius)
    image = BoxSet(P, Set{Int32}())
    while !isnothing(keys.nextvalstate)
        in_keys = CuArray{Int32,1}(collect(take(keys, SZ)))
        nk, np = Int32(length(in_keys)), Int32(length(points))
        n = nk * np
        out_keys = CuArray{Int32,1}(undef, n)
        args = (g.map, in_keys, points, out_keys, P, nk, np)
        kernel! = @cuda launch=false map_boxes_kernel!(args...)
        config  = launch_configuration(kernel!.fun)
        threads = min(n, config.threads)
        blocks  = cld(n, threads)
        kernel!(args...; threads, blocks)
        CUDA.synchronize()
        out_cpu = Array{Int32,1}(out_keys)
        y = Set(out_cpu)
        delete!(y, 0i32)
        union!(image, BoxSet(P, y))
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    return image
end
