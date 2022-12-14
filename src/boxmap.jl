"""
    BoxMap(map, domain; no_of_points=ntuple(_->4, N)) -> SampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `SampledBoxMap` defined 
on `Box`es. By default uses a grid of sample points. 

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
BoxMap(symb::Symbol, args...; kwargs...) = BoxMap(Val(symb), args...; kwargs...)

# default BoxMap behavior
BoxMap(args...; kwargs...) = GridBoxMap(args...; kwargs...)

for str in (
        "PointDiscretized", 
        "MonteCarlo", 
        "Grid", 
        "Adaptive", 
        "Sampled", 
        "CPUSampled", 
        "GPUSampled", 
        "Interval"
    )

    @eval BoxMap(::Val{Symbol($str)}, args...; kwargs...) = $(Symbol(str*"BoxMap"))(args...; kwargs...)
    @eval BoxMap(::Val{Symbol(lowercase($str))}, args...; kwargs...) = $(Symbol(str*"BoxMap"))(args...; kwargs...)

end

BoxMap(c::Val{:cpu}, args...; kwargs...) = GridBoxMap(c, args...; kwargs...)
BoxMap(c::Val{:gpu}, args...; kwargs...) = GridBoxMap(c, args...; kwargs...)

Base.show(io::IO, g::BoxMap) = print(io, "BoxMap over $(g.domain)")
