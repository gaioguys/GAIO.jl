"""
    BoxMap(map, domain; no_of_points=ntuple(_->4, N)) -> SampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `SampledBoxMap` defined 
on `Box`es. 

By default uses adaptive test-point sampling. 
For SIMD- and GPU-accelerated `BoxMap`s, uses
a grid of test points by default. 
"""
BoxMap(symb::Symbol, args...; kwargs...) = BoxMap(Val(symb), args...; kwargs...)
BoxMap(symb::Symbol, accel::Symbol, args...; kwargs...) = BoxMap(symb, Val(accel), args...; kwargs...)

# default BoxMap behavior
BoxMap(args...; kwargs...) = BoxMap(:adaptive, args...; kwargs...)

for str in (
        "PointDiscretized", 
        "MonteCarlo", 
        "Grid", 
        "Adaptive", 
        "Sampled", 
        "Interval"
    )

    @eval BoxMap(::Val{Symbol($str)}, args...; kwargs...) = $(Symbol(str*"BoxMap"))(preprocess(args...)...; kwargs...)
    @eval BoxMap(::Val{Symbol(lowercase($str))}, args...; kwargs...) = $(Symbol(str*"BoxMap"))(preprocess(args...)...; kwargs...)

end

Base.show(io::IO, g::BoxMap) = print(io, "BoxMap over $(g.domain)")

# this is a no-op outside of extensions
preprocess(args...) = args

function (g::BoxMap)(source::BoxSet; show_progress::Bool=false, kwargs...) 
    map_boxes(g, source, Val(show_progress); kwargs...)
end

function map_boxes(g::BoxMap, source::BoxSet, show_progress::Val{false}; kwargs...)
    map_boxes(g, source; kwargs...)
end

function map_boxes(g::BoxMap, source::BoxSet, show_progress::Val{true}; kwargs...)
    @error "Progress meter code not loaded. Run `using ProgressMeter` to get progress meters."
end
