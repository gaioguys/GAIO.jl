struct BoxMapGPUCache{P,Q}
    bools::P
    sums::Q
end

for T in (:Float, :Int, :UInt, :ComplexF), len in (:16, :64)
    ti, to = Symbol(T, len), Symbol(T, :32)
    @eval convertto32(x::$ti) = $to(x)
end
convertto32(x::V) where {N,T,V<:SVNT{N,T}} = map(convertto32, x)
convertto32(x) = x
#convertto32(s::BoxPartition) = BoxPartition(convertto32.((s.domain, s.left, s.scale, s.dims, s.dimsprod)))
#convertto32(s::BoxSet) = BoxSet(convertto32(s.partition), convertto32(s.set))

function PointDiscretizedMap(map, domain, points::V, ::Val{:gpu}) where {N,T,V<:AV{<:SVNT{N,T}}}
    points_vec = CuArray(convertto32.(points))
    bools = CuArray{Bool,1}(undef, N)
    sums  = CuArray{UInt32,1}(undef, N)
    return PointDiscretizedMap(map, domain, points_vec, BoxMapGPUCache(bools, sums))
end

@muladd function map_boxes_kernel!(
        G::Function, bools, keys, points, 
        P::BoxPartition, np, nk
    )

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
        if isnothing(hit)
            continue
        end
        bools[hit] = true
    end
end

function findall_kernel!(out, bools, sums)
    T = eltype(out)
    ind = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in ind:stride:length(sums)
        if bools[i]
            b = sums[i]
            out[b] = T(ind)
        end
    end
end

function launch_kernel_sync!(len, func, args...)
    kernel! = @cuda launch=false func(args...)
    config  = launch_configuration(kernel!.fun)
    threads = min(len, config.threads)
    blocks  = cld(len, threads)
    kernel!(args...; threads, blocks)
    CUDA.synchronize()
    return
end

function map_boxes(
        g::SampledBoxMap{<:BoxMapGPUCache,N}, 
        source::BoxSet{BoxPartition{N,T,I,D}}
    ) where {N,T,I,D}

    P, bools, sums = source.partition, g.acceleration.bools, g.acceleration.sums
    lP = length(P)
    if length(bools) != lP || length(sums) != lP
        resize!(bools, lP)
        resize!(sums, lP)
    end
    CUDA.fill!(bools, false)
    CUDA.fill!(sums, 0ui32)
    keys = CuArray(collect(source.set))
    points = g.domain_points(P.domain.center, P.domain.radius)
    nk, np = I(length(keys)), I(length(points))
    args = (g.map, bools, keys, points, P, nk, np)
    launch_kernel_sync!(nk * np, map_boxes_kernel!, args...)
    accumulate!(+, sums, bools); CUDA.synchronize()
    n = CUDA.@allowscalar sums[end]
    if n > 0ui32
        y = CuArray{I}(undef, n)
        launch_kernel_sync!(lP, findall_kernel!, y, bools, sums)
        boxset = BoxSet(P, Set{I}(Array(y)))
        CUDA.unsafe_free!(y)
    else
        boxset = boxset_empty(P)
    end
    CUDA.unsafe_free!(keys)
    return boxset
end
