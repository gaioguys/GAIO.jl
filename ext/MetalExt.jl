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

# Constructors are copied from CUDAExt
"""
    BoxMap(:gpu, map, domain; n_points) -> GPUSampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `GPUSampledBoxMap` defined 
on `Box`es. 

Uses the GPU's acceleration capabilities. 

By default uses a grid of sample points. 


    BoxMap(:montecarlo, :gpu, boxmap_args...)
    BoxMap(:grid, :gpu, boxmap_args...)
    BoxMap(:pointdiscretized, :gpu, boxmap_args...)
    BoxMap(:sampled, :gpu, boxmap_args...)

Type representing a dicretization of a map using 
sample points, which are mapped on the gpu. This 
type performs orders of magnitude faster than 
standard `SampledBoxMap`s when point mapping is the 
bottleneck. 

!!! warning "`image_points` with `GPUSampledBoxMap`"
    `GPUSampledBoxMap` makes NO use of the `image_points` 
    field in `SampledBoxMap`s. 

Fields:
* `map`:              map that defines the dynamical system.
* `domain`:           domain of the map, `B`.
* `domain_points`:    the spread of test points to be mapped forward 
                      in intersection algorithms.
                      WARNING: this differs from 
                      SampledBoxMap.domain_points in that it is a vector 
                      of "global" test points within [-1, 1]ᴺ. 


Requires a Metal-capable gpu. 
"""
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

function GPUSampledBoxMap(boxmap::SampledBoxMap{N}) where {N}
    c, r = boxmap.domain
    points = boxmap.domain_points(c, r)
    map!(x -> (x .- c) ./ r, points, points)    # send all points to [-1, 1]ᴺ
    GPUSampledBoxMap(boxmap.map, boxmap.domain, mtl(points))
end

function BoxMap(symb::Symbol, ::Val{:gpu}, args...; kwargs...)
    F = BoxMap(symb, args...; kwargs...)
    GPUSampledBoxMap(F)
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


# helper + compatibility functions are almost entirely copied from CUDAExt.
# This amount of code-copying is unfortunate, though I don't know of a good 
# way to get around it
function Base.show(io::IO, g::GPUSampledBoxMap)
    n = length(g.boxmap.domain_points(g.boxmap.domain...).iter)
    print(io, "GPUSampledBoxMap with $(n) sample points")
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
