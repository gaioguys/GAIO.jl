struct BoxMapGPUCache{SZ}
    maxsize::Val{SZ}
end

for T in (:Float, :Int, :UInt, :ComplexF), len in (:16, :64)
    ti, to = Symbol(T, len), Symbol(T, :32)
    @eval convertto32(x::$ti) = $to(x)
end
convertto32(x::V) where {N,T,V<:SVNT{N,T}} = map(convertto32, x)
convertto32(x::Box) = Box(map(convertto32, (x.center, x.radius))...)
convertto32(x::BoxPartition) = BoxPartition(map(convertto32, (x.domain, x.left, x.scale, x.dims, x.dimsprod))...)
convertto32(x) = x

for type in (:(BoxPartition{N,Float64,J}), :(BoxPartition{N,J,Int64}), :(BoxPartition{N,J,Int128}))
    @eval function map_boxes(g::SampledBoxMap{<:BoxMapGPUCache}, source::BoxSet{$type}) where {N,J}
        return map_boxes(g, BoxSet(convertto32(source.partition), Set{Int32}(source.set)))
    end
end

function PointDiscretizedMap(map, domain, points, ::Val{:gpu})
    points_vec = CuArray(convertto32.(points))
    maxsize = min(prod(CUDA.max_grid_size), CUDA.totalmem(CUDA.device()))
    maxsize = maxsize ÷ sizeof(Int32) ÷ length(points_vec) ÷ 2 ÷ 5 * 4
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

function map_boxes(
        g::SampledBoxMap{<:BoxMapGPUCache{SZ}}, 
        source::BoxSet{BoxPartition{N,T,I,D}}
    ) where {N,T,I,D,SZ}

    P, keys = source.partition, Stateful(source.set)
    points = g.domain_points(P.domain.center, P.domain.radius)
    set = BoxSet(P, Set{Int32}())
    while !isnothing(keys.nextvalstate)
        in_keys = CuArray{I,1}(collect(take(keys, SZ)))
        nk, np = I(length(in_keys)), I(length(points))
        n = nk * np
        out_keys = CuArray{Int32,1}(undef, n)
        args = (g.map, in_keys, points, out_keys, P, nk, np)
        kernel! = @cuda launch=false map_boxes_kernel!(args...)
        config  = launch_configuration(kernel!.fun)
        threads = min(n, config.threads)
        blocks  = cld(n, threads)
        kernel!(args...; threads, blocks)
        CUDA.synchronize()
        out_cpu = Array(out_keys)
        y = Set(out_cpu)
        delete!(y, 0i32)
        union!(set, BoxSet(P, y))
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    return set
end
