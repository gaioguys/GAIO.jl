"""
    SmartBoxMap(adaptive, montecarlo)

Type represtinging a discretization of a map 
which attempts to construct rigorous coverings 
of map images using `sampled_adaptive` and 
constructs transfers for `TransferOperator`s 
using Monte-Carlo sample points. 
"""
struct SmartBoxMap{F,G} <: BoxMap
    adaptive::F
    montecarlo::G
end

map_boxes(g::SmartBoxMap, source::BoxSet) = map_boxes(g.adaptive, source)
construct_transfers(g::SmartBoxMap, source::BoxSet) = construct_transfers(g.montecarlo, source)

Core.@doc raw"""
    BoxMap(map, domain; no_of_points=64*N) -> SmartBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `SampledBoxMap` defined 
on `Box`es. Attempts to use adaptive sampling to 
construct rigorous coverings of map images and uses 
Monte-Carlo sample points to construct 
`TransferOperator`s. 

Specific Constructors:
* `BoxMap`
* `PointDiscretizedBoxMap`
* `MonteCarloBoxMap`
* `GridBoxMap`
* `AdaptiveBoxMap`
* `CPUSampledBoxMap`
* `GPUSampledBoxMap`
* `IntervalBoxMap`
"""
function BoxMap(f, domain::Box{N,T}; no_of_points=4*N*pick_vector_width(T)) where {N,T}

    montecarlo_points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:no_of_points ]
        
    function try_sample_adaptive(center::SVNT{N,T}, radius::SVNT{N,T}) where {N,T}
        try 
            points = sample_adaptive(f, center, radius)
        catch ex
            ex isa Union{DomainError, LAPACKException} || rethrow(ex)
            @warn("""
                An Exception was thrown during adaptive sampling. 
                Using Monte-Carlo sample points.
                """, 
                exception=ex,
                box=Box{N,T}(center, radius)
            )
            points = rescale(center, radius, montecarlo_points)
        end
    end 

    adaptive = SampledBoxMap(f, domain, try_sample_adaptive, vertices)
    montecarlo = SampledBoxMap(f, domain, rescale(montecarlo_points), center)

    SmartBoxMap(adaptive, montecarlo)
end

function BoxMap(f, P::BoxPartition{N,T}; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    BoxMap(f, P.domain, no_of_points=no_of_points)
end

Base.show(io::IO, g::BoxMap) = print(io, "BoxMap over $(g.domain)")
Base.show(io::IO, g::SmartBoxMap) = print(io, "BoxMap over $(g.adaptive.domain)")
