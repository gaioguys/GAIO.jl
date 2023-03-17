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
BoxMap(accel::Val{:simd}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)
BoxMap(accel::Val{:gpu}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)

# default BoxMap behavior
BoxMap(args...; kwargs...) = AdaptiveBoxMap(args...; kwargs...)

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

Base.show(io::IO, g::BoxMap) = print(io, "BoxMap over $(g.domain)")
