module CUDAExt

using GAIO, CUDA, StaticArrays, MuladdMacro

import Base.Iterators: Stateful, take
import Base: unsafe_trunc
import Base: @propagate_inbounds
import CUDA: Adapt
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, point_to_key, ⊔, SVNT
import GAIO: @common_gpu_code


@common_gpu_code "CUDA" cu CuArray CUDA.CuArrayKernelAdaptor CUDA.KernelAdaptor


@muladd function map_boxes_kernel!(g, P, domain_points, in_keys, out_keys)
    nk = length(in_keys)*i32
    np = length(domain_points)*i32
    ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x #- 1i32
    stride = gridDim().x * blockDim().x
    len = nk * np #- 1i32
    for i in ind : stride : len
        m, n = CartesianIndices((np, nk))[i].I
        key  = in_keys[n]
        box  = key_to_box(P, key)
        c, r = box
        p    = domain_points[m]
        p    = @. c + r * p
        fp   = typesafe_map(P, g, p)
        hit  = @inbounds point_to_key(P, fp, Val(:gpu))
        out_keys[i] = isnothing(hit) ? out_of_bounds(P) : hit
    end
end

function map_boxes(
        G::GPUSampledBoxMap{N,T}, source::BoxSet{B,Q,S}
    ) where {N,T,B,Q<:BoxGrid,S}

    g = G.boxmap
    p = g.domain_points(g.domain...).iter
    np = length(p)
    keys = Stateful(source.set)
    P = source.partition
    K = cu_reduce(keytype(Q))
    image = S()
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_cpu = collect(K, take(keys, stride))
        in_keys = CuArray{K,1}(in_cpu)
        nk = length(in_cpu)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        union!(image, out_cpu)
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(image, out_of_bounds(P))
    return BoxSet(P, image)
end

function construct_transfers(
        G::GPUSampledBoxMap, domain::BoxSet{R,Q,S}, codomain::BoxSet{U,H,W}
    ) where {N,T,R<:Box{N,T},Q,S,U,H,W}

    g = G.boxmap
    p = g.domain_points(g.domain...).iter
    np = length(p)
    keys = Stateful(domain.set)
    P = domain.partition
    P2 = codomain.partition
    P == P2 || throw(DomainError((P, P2), "Partitions of domain and codomain do not match. For GPU acceleration, they must be equal."))
    K = cu_reduce(keytype(Q))
    D = Dict{Tuple{K,K},cu_reduce(T)}
    mat = D()
    codomain = BoxSet(P2, S())
    oob = out_of_bounds(P)
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_cpu = collect(K, take(keys, stride))
        in_keys = CuArray{K,1}(in_cpu)
        nk = length(in_cpu)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        C = CartesianIndices((np, nk))
        for i in 1:nk*np
            _, n = C[i].I
            key, hit = in_cpu[n], out_cpu[i]
            hit == oob && continue
            hit in codomain.set || continue
            mat = mat ⊔ ((hit,key) => 1)
        end
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    return mat
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


end # module
