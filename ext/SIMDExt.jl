module SIMDExt
    
using GAIO, SIMD, HostCPUFeatures, MuladdMacro, Base.Threads, StaticArrays, FLoops

import Base: @propagate_inbounds
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, ⊔, SVNT

#export CPUSampledBoxMap

"""
    BoxMap(:cpu, map, domain; n_points) -> CPUSampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `CPUSampledBoxMap` defined 
on `Box`es. 

Uses the CPU's SIMD acceleration capabilities. 

By default uses a grid of sample points. 


    BoxMap(:sampled, :cpu, boxmap, idx_base)

Type representing a discretization of a map using 
sample points which are explicitly vectorized. This 
type performs roughly 2x as many floating point 
operations per second as standard `SampledBoxMap`s. 

Fields:
* `boxmap`:         `SampledBoxMap` with one restriction:
                    `boxmap.domain_points(c, r)` must 
                    return an iterable with eltype 
                    `SVector{N, SIMD.Vec{S,T}}` where `N`
                    is the dimension, `S` is the cpu's 
                    SIMD operation capacity, e.g. `4`, 
                    and `T` is the individual element type, 
                    e.g. `Float64`. 
* `idx_base`:       `SIMD.Vec{S,Int}` which is used to 
                    transform a 
                    `Vector{SVector{N, SIMD.Vec{S,T}}}`
                    into a 
                    `Vector{SVector{N,T}}`. 

.
"""
struct CPUSampledBoxMap{simd,N,T,F<:SampledBoxMap{N,T}} <: BoxMap
    boxmap::F
    idx_base::SIMD.Vec{simd,Int}
end

# Constuctors

BoxMap(::Val{Symbol("CPUSampled")}, args...; kwargs...) = CPUSampledBoxMap(args...; kwargs...)
BoxMap(::Val{Symbol("cpusampled")}, args...; kwargs...) = CPUSampledBoxMap(args...; kwargs...)
BoxMap(accel::Val{:simd}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)

function CPUSampledBoxMap(boxmap::SampledBoxMap{N,T}) where {N,T}
    simd = Int(pick_vector_width(T))
    idx_base = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    CPUSampledBoxMap(boxmap, idx_base)
end

"""
    BoxMap(:pointdiscretized, :simd, map, domain, points) -> CPUSampledBoxMap

Construct a `CPUSampledBoxMap` that uses the iterator 
`points` as test points. `points` must have eltype 
`SVector{N, SIMD.Vec{S,T}}` and be within the unit 
cube `[-1,1]^N`. 
"""
function PointDiscretizedBoxMap(::Val{:simd}, map, domain::Box{N,T}, points) where {N,T}
    n, simd = length(points), Int(pick_vector_width(T))
    if n % simd != 0
        throw(DimensionMismatch("Number of test points $n is not divisible by $T SIMD capability $simd"))
    end
    gathered_points = tuple_vgather(points, simd)
    domain_points = rescale(gathered_points)
    image_points = center
    boxmap = SampledBoxMap(map, domain, domain_points, image_points)
    CPUSampledBoxMap(boxmap)
end

function PointDiscretizedBoxMap(c::Val{:simd}, map, P::Q, points) where {N,T,Q<:AbstractBoxPartition{Box{N,T}}}
    PointDiscretizedBoxMap(c, map, P.domain, points)
end

"""
    BoxMap(:grid, :simd, map, domain::Box{N}; n_points::NTuple{N} = ntuple(_->16, N)) -> CPUSampledBoxMap

Construct a `CPUSampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `n_points`, which is 
a tuple of length equal to the dimension of the domain. 
The number of points is rounded up to the nearest mutiple 
of the cpu's SIMD capacity. 
"""
function GridBoxMap(c::Val{:simd}, map, domain::Box{N,T}; n_points=ntuple(_->4,N)) where {N,T}
    simd = Int(pick_vector_width(T))
    n_points = ntuple(N) do i
        rem = n_points[i] % simd
        n_points[i] + (rem == 0 ? rem : simd - rem)
    end
    Δp = 2 ./ n_points
    points = SVector{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(n_points) ]
    PointDiscretizedBoxMap(c, map, domain, points)
end

function GridBoxMap(c::Val{:simd}, map, P::Q; n_points=ntuple(_->4,N)) where {N,T,Q<:AbstractBoxPartition{Box{N,T}}}
    GridBoxMap(c, map, P.domain; n_points=n_points)
end

"""
    BoxMap(:montecarlo, :simd, map, domain::Box{N}; n_points=16*N) -> SampledBoxMap

Construct a `CPUSampledBoxMap` that uses `n_points` 
Monte-Carlo test points. The number of points is rounded 
up to the nearest multiple of the cpu's SIMD capacity. 
"""
function MonteCarloBoxMap(c::Val{:simd}, map, domain::Box{N,T}; n_points=16*N) where {N,T}
    simd = Int(pick_vector_width(T))
    rem = n_points % simd
    n_points = n_points + (rem == 0 ? rem : simd - rem)
    points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:n_points ] 
    PointDiscretizedBoxMap(c, map, domain, points)
end 

function MonteCarloBoxMap(c::Val{:simd}, map, P::Q; n_points=16*N) where {N,T,Q<:AbstractBoxPartition{Box{N,T}}}
    MonteCarloBoxMap(c, map, P.domain; n_points=n_points)
end

function Base.show(io::IO, G::CPUSampledBoxMap{simd}) where {simd}
    g = G.boxmap
    center, radius = g.domain
    n = length(g.domain_points(center, radius)) * simd
    print(io, "CPUSampledBoxMap with $(n) sample points")
end

function typesafe_map(g::CPUSampledBoxMap{simd,N,T}, x::SVNT{N}) where {simd,N,T}
    convert(SVector{N,SIMD.Vec{simd,T}}, g.boxmap.map(x))
end

# BoxMap API

@inbounds function map_boxes(
        G::CPUSampledBoxMap{simd,N,T}, source::BoxSet{B,Q,S}
    ) where {simd,N,T,B,Q,S}

    P = source.partition
    g, idx_base = G
    @floop for box in source
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        c, r = box
        for p in g.domain_points(c, r)
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P, ip)
                    @reduce() do (image = S(); hit)     # Initialize empty key set
                        image = image ⊔ hit             # Add hit key to image
                    end
                end
            end
        end
    end
    return BoxSet(P, image::S)
end

@inbounds function construct_transfers(
        G::CPUSampledBoxMap{simd}, domain::BoxSet{R,Q,S}
    ) where {simd,N,T,R<:Box{N,T},Q,S}
    
    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}
    g, idx_base = G

    @floop for key in keys(domain)
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        box = key_to_box(P, key)
        c, r = box
        for p in g.domain_points(c, r)
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P, ip)
                    weight = (hit,key) => one(T)
                    @reduce() do (image = S(); hit), (mat = D(); weight)    # Initialize empty key set and dict-of-keys sparse matrix
                        image = image ⊔ hit                                 # Add hit key to image
                        mat = mat ⊔ weight                                  # Add weight to mat[hit, key]
                    end
                end
            end
        end
    end

    codomain = BoxSet(P, image::S)
    return mat::D, codomain
end

@inbounds @muladd function construct_transfers(
        G::CPUSampledBoxMap{simd}, domain::BoxSet{R,Q}, codomain::BoxSet{U,H}
    ) where {simd,N,T,R<:Box{N,T},Q,U,H}

    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}
    g, idx_base = G

    @floop for key in keys(domain)
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        box = key_to_box(P1, key)
        c, r = box
        for p in g.domain_points(c, r)
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P2, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P2, ip)
                    hit in codomain.set || continue
                    weight = (hit,key) => one(T)
                    @reduce() do (mat = D(); weight)     # Initialize dict-of-keys sparse matrix
                        mat = mat ⊔ weight               # Add weight to mat[hit, key]
                    end
                end
            end
        end
    end
    return mat::D
end

# helper + compatibility functions

Base.iterate(c::CPUSampledBoxMap, i...) = (c.boxmap, Val(:idx_base))
Base.iterate(c::CPUSampledBoxMap, ::Val{:idx_base}) = (c.idx_base, Val(:done))
Base.iterate(c::CPUSampledBoxMap, ::Val{:done}) = nothing

@propagate_inbounds function tuple_vgather(
        v::V, idx::SIMD.Vec{simd,Int}# = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    ) where {N,T,simd,V<:AbstractArray{<:SVNT{N,T}}}

    vr = reinterpret(T,v)
    vo = ntuple(i -> vr[idx + i], Val(N))
    return vo
end

@propagate_inbounds @muladd function tuple_vgather(
        v::V, simd::Integer
    ) where {N,T,V<:AbstractArray{<:SVNT{N,T}}}

    n = length(v)
    m = n ÷ simd
    @boundscheck if n != m * simd
        throw(DimensionMismatch("length of input ($n) % simd ($simd) != 0"))
    end
    vr = reinterpret(T, v)
    vo = Vector{SVector{N,SIMD.Vec{simd,T}}}(undef, m)#ntuple(i -> vr[idx + i], Val(N))
    idx = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    for i in 1:m
        vo[i] = ntuple( j -> vr[idx + (i-1)*N*simd + j], Val(N) )
    end
    return vo
end

@propagate_inbounds function tuple_vgather_lazy(
        v::V, simd
    ) where {N,T,V<:AbstractArray{<:SVNT{N,T}}}
    
    n = length(v)
    m = n ÷ simd
    @boundscheck if n != m * simd
        throw(DimensionMismatch("length of input ($n) % simd ($simd) != 0"))
    end
    vr = v |>
        x -> reinterpret(T, v) |> 
        x -> reshape(x, (N,simd,m)) |> 
        x -> PermutedDimsArray(x, (2,1,3)) |> 
        x -> reshape(x, (N*n,)) |>
        x -> reinterpret(SVector{N,SIMD.Vec{simd,T}}, x)
    return vr
end

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI, idx::SIMD.Vec{simd,I}
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:SVNT{N,SIMD.Vec{simd,T}},I<:Integer}
    
    for i in 1:N
        vo[idx + i] = vi[i]
    end
    return vo
end

@propagate_inbounds @muladd function tuple_vscatter!(
        vo::VO, vi::VI
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:AbstractArray{<:SVNT{N,SIMD.Vec{simd,T}}}}

    idx = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    for j in 1:length(vi)
        tuple_vscatter!( vo, vi[j], idx + (j-1)*N*simd )
    end
    return vo
end

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI, idx::SIMD.Vec{simd,I}
    ) where {N,T,simd,VO<:AbstractArray{<:SVNT{N,T}},VI<:SVNT{N,SIMD.Vec{simd,T}},I<:Integer}

    vr = reinterpret(T, vo)
    return tuple_vscatter!(vr, vi, idx)
end

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI
    ) where {N,T,simd,VO<:AbstractArray{<:SVNT{N,T}},VI<:AbstractArray{<:SVNT{N,SIMD.Vec{simd,T}}}}
    
    vr = reinterpret(T, vo)
    return tuple_vscatter!(vr, vi)
end

end # module
