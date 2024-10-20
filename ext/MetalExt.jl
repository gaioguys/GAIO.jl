module MetalExt
    
using GAIO, Metal, StaticArrays, MuladdMacro
using ThreadsX

#import Base.Iterators: Stateful, take
import Base: unsafe_trunc
import Base: @propagate_inbounds, @boundscheck
import Metal: Adapt
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, point_to_key, ⊔, SVNT

#export GPUSampledBoxMap

BoxMap(::Val{Symbol("GPUSampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(::Val{Symbol("gpusampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(accel::Val{:gpu}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)
    
struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

struct GPUSampledBoxMap{N,T,F<:Function} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::MtlArray{SVector{N,T}}

    function GPUSampledBoxMap{N,T,F}(map::F, domain::Box, domain_points::MtlArray{SVector{N,T}}) where {N,T,F<:Function}
        new{N,T,F}(map, Box{N,T}(domain...), domain_points)
    end
end

function GPUSampledBoxMap(map::F, domain::Box, domain_points::MtlArray{SVector{N,T}}) where {N,T,F<:Function}
    GPUSampledBoxMap{N,T,F}(map, domain, domain_points)
end

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

function map_boxes(G::GPUSampledBoxMap{F}, source::BoxSet{B,Q,S}) where {B,Q,S,F}
    P = mtl(source.partition)
    _, cpu_keys = execute_boxmap(G, source)
    image = BoxSet(P, S())
    union!(image.set, ThreadsX.Set(cpu_keys))
    delete!(image.set, out_of_bounds(P))
    return image
end

function construct_transfers(
        G::GPUSampledBoxMap{F}, domain::BoxSet{R,_Q,S}#, codomain::BoxSet{U,_H,W}
    ) where {N,T,R<:Box{N,T},_Q,S,F}

    P = domain.partition
    codomain = BoxSet(P, S())
    P = mtl(P)

    in_cpu, out_cpu = execute_boxmap(G, domain)

    oob = out_of_bounds(P)
    Q = typeof(P)
    K = keytype(Q)

    union!(codomain.set, out_cpu)
    delete!(codomain.set, oob)

    mat = Dict{Tuple{K,K},cu_reduce(T)}()
    for cartesian_ind in CartesianIndices(out_cpu)
        i, j = cartesian_ind.I
        key = in_cpu[i]
        hit = out_cpu[i,j]
        checkbounds(Bool, P, hit) || continue # check for oob
        mat = mat ⊔ ((hit,key) => 1)
    end

    return mat, codomain
end

function construct_transfers(
        G::GPUSampledBoxMap{F}, domain::BoxSet{R,_Q,S}, codomain::BoxSet{U,_H,W}
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

    #@info "kernel initialization" n_keys n_samples n_keys_per_group n_groups leftover

    @metal threads=(n_keys_per_group,n_samples) groups=n_groups map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, 0i32)

    if leftover > 0
        @metal threads=(leftover,n_samples) groups=1 map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, leftover*i32)
    end

    Metal.synchronize()

    return Array(in_keys), Array(out_keys)
end

# constructors
"""
    BoxMap(:pointdiscretized, :gpu, map, domain::Box{N}, points) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses the Vector `points` as test points. 
`points` must be a VECTOR of test points within the unit cube 
`[-1,1]^N`. 

Requires a Metal-capable gpu. 
"""
function PointDiscretizedBoxMap(::Val{:gpu}, map, domain::Box{N,T}, points) where {N,T}
    domain_points = MtlArray{SVector{N,T}}(points)
    GPUSampledBoxMap(map, domain, domain_points)
end

"""
    BoxMap(:grid, :gpu, map, domain::Box{N}; n_points::NTuple{N} = ntuple(_->16, N)) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `n_points`, which is 
a tuple of length equal to the dimension of the domain. 

Requires a Metal-capable gpu. 
"""
function GridBoxMap(::Val{:gpu}, map, domain::Box{N,T}; n_points=ntuple(_->4,N)) where {N,T}
    Δp = 2 ./ n_points
    points = SVector{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(n_points) ]
    GPUSampledBoxMap(map, domain, mtl(points))
end

function GridBoxMap(c::Val{:gpu}, map, P::BoxGrid{N,T}; n_points=ntuple(_->4,N)) where {N,T}
    GridBoxMap(c, map, P.domain, n_points=n_points)
end

"""
    BoxMap(:montecarlo, :gpu, map, domain::Box{N}; n_points=16*N) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses `n_points` 
Monte-Carlo test points. 

Requires a Metal-capable gpu. 
"""
function MonteCarloBoxMap(::Val{:gpu}, map, domain::Box{N,T}; n_points=16*N) where {N,T}
    points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:n_points ] 
    GPUSampledBoxMap(map, domain, mtl(points))
end 

function MonteCarloBoxMap(c::Val{:gpu}, map, P::BoxGrid{N,T}; n_points=16*N) where {N,T}
    MonteCarloBoxMap(c, map, P.domain; n_points=n_points)
end

# helper + compatibility functions are almost entirely copied from CUDAExt.
# This amount of code-copying is unfortunate, though I don't know of a good 
# way to get around it

function typesafe_map(::Q, g, p) where {N,T,B<:Box{N,T},Q<:BoxLayout{B}}
    SVector{N,T}( g(p) )
end

# hotfix to avoid errors due to cuda device-side printing
@propagate_inbounds function point_to_key(partition::BoxGrid{N,T,I}, point, ::Val{:gpu}) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, partition, x_ints)
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(partition))
    end
    return x_ints
end

function out_of_bounds(::P) where {N,T,I,P<:BoxGrid{N,T,I}}
    K = cu_reduce(keytype(P))
    K(ntuple(_->0, Val(N)))
end

function out_of_bounds(::P) where {N,T,I,P<:BoxTree{N,T,I}}
    K = cu_reduce(keytype(P))
    K((0, ntuple(_->0, Val(N))))
end

function Adapt.adapt_storage(::A, (c,r)::Box{N,T}) where {N,T,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT = cu_reduce(T)
    Box{N,TT}(c,r)
end

function Adapt.adapt_structure(a::A, b::BoxGrid{N,T,I}) where {N,T,I,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT, II = cu_reduce(T), cu_reduce(I)
    Adapt.adapt_storage(a, 
        BoxGrid{N,TT,II}(
            Box{N,TT}(b.domain...),
            SVector{N,TT}(b.left),
            SVector{N,TT}(b.scale),
            SVector{N,II}(b.dims)
        )
    )
end

function Adapt.adapt_structure(a::A, b::BoxTree{N,T,I}) where {N,T,I,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT = cu_reduce(T)
    Adapt.adapt_storage(a,
        BoxTree(
            Box{N,TT}(b.domain...),
            Adapt.adapt(a, b.nodes)
        )
    )
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F},M}}

    MtlArray{SVector{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,F<:Integer,V<:AbstractArray{<:NTuple{N,F},M}}

    MtlArray{NTuple{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,H<:Integer,F<:Integer,V<:AbstractArray{<:Tuple{H,NTuple{N,F}},M}}

    MtlArray{Tuple{H,NTuple{N,cu_reduce(F)}},M}(x)
end

cu_reduce(::Type{I}) where {I<:Integer} = Int32
cu_reduce(::Type{Int16}) = Int16
cu_reduce(::Type{Int8}) = Int8
cu_reduce(::Type{F}) where {F<:AbstractFloat} = Float32
cu_reduce(::Type{Float16}) = Float16
cu_reduce(::Type{<:NTuple{N,T}}) where {N,T} = NTuple{N,cu_reduce(T)}

end # module
