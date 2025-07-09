module MetalExt
    
using GAIO, Metal, StaticArrays, MuladdMacro
#using ThreadsX

#import Base.Iterators: Stateful, take
import Base: unsafe_trunc
import Base: @propagate_inbounds, @boundscheck
import Metal: Adapt
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, point_to_key, ⊔, SVNT
import GAIO: @common_gpu_code


@common_gpu_code "Metal" mtl MtlArray Metal.MtlArrayAdaptor Metal.Adaptor


@muladd function map_boxes_kernel!(g, P::Q, in_keys, out_keys, domain_points, offset) where {N,T,B<:Box{N,T},Q<:BoxLayout{B}}
    x, y = thread_position_in_grid_2d()
    x += offset
    k = in_keys[x]
    c, r = key_to_box(P, k)
    p = domain_points[y]
    p = @. c + r * p
    p = typesafe_map(P, g, p)
    hit = point_to_key(P, p, Val(:gpu))
    out_keys[x,y] = isnothing(hit) ? out_of_bounds(P) : hit
    return nothing
end

function map_boxes(
        G::GPUSampledBoxMap{F}, source::BoxSet{B,Q,S}; 
        show_progress=false     # does nothing here
    ) where {B,Q,S,F}

    P = mtl(source.partition)
    _, cpu_keys = execute_boxmap(G, source)
    image = BoxSet(P, S())
    union!(image.set, cpu_keys)    # union!(image.set, ThreadsX.Set(cpu_keys))
    delete!(image.set, out_of_bounds(P))
    return image
end

function construct_transfers(
        G::GPUSampledBoxMap{F}, domain::BoxSet{R,_Q,S}, codomain::BoxSet{U,_H,W}; 
        show_progress=false     # does nothing here
    ) where {N,T,R<:Box{N,T},_Q,S,U,_H,W,F}

    P = mtl(domain.partition)
    P2 = mtl(codomain.partition)
    P == P2 || throw(DomainError((P, P2), "Partitions of domain and codomain do not match. For GPU acceleration, they must be equal."))

    in_cpu, out_cpu = execute_boxmap(G, domain)

    Q = typeof(P)
    K = keytype(Q)

    mat = Dict{Tuple{K,K},cu_reduce(T)}()
    for cartesian_ind in CartesianIndices(out_cpu)
        i, j = cartesian_ind.I
        key = in_cpu[i]
        hit = out_cpu[i,j]
        checkbounds(Bool, P, hit) || continue # check for oob
        hit in codomain.set || continue
        mat = mat ⊔ ((hit,key) => 1)
    end

    return mat
end

function execute_boxmap(G::GPUSampledBoxMap, source::BoxSet)
    g = G.map;  domain_points = G.domain_points
    n_samples = length(G.domain_points)

    in_cpu = collect(source.set)
    in_keys = mtl(in_cpu)
    n_keys = length(in_keys)
    
    P = mtl(source.partition)
    Q = typeof(P)
    out_keys = mtl(Array{keytype(Q)}( undef, (n_keys,n_samples) ))

    n_keys_per_group = min(1024 ÷ n_samples, n_keys)
    n_groups = n_keys ÷ n_keys_per_group

    leftover = n_keys % (n_groups * n_keys_per_group)

    @debug "kernel initialization" n_keys n_samples n_keys_per_group n_groups leftover

    @metal threads=(n_keys_per_group,n_samples) groups=n_groups map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, 0i32)

    if leftover > 0
        @metal threads=(leftover,n_samples) groups=1 map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, leftover*i32)
    end

    Metal.synchronize()

    return Array(in_keys), Array(out_keys)
end


end # module
