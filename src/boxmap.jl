"""
    BoxMap(map, domain) -> SampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `SampledBoxMap` defined 
on `Box`es. 

By default uses adaptive test-point sampling. 
For GPU-accelerated `BoxMap`s, uses
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

function (g::BoxMap)(source::BoxSet; kwargs...) 
    map_boxes(g, source; kwargs...)
end
