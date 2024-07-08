const F = (Int === Int64) ? Float64 : Float32

Core.@doc raw"""
    Box{N,T}(center, radius)
    Box(center, radius)

A generalized box in dimension `N` with element type `T`. 
Mathematically, this is a set
```math
[center_1 - radius_1,\ center_1 + radius_1) \ \times \ \ldots \ \times \ [center_N - radius_N,\ center_N + radius_N)
```

Fields:
* `center`:   vector where the box's center is located
* `radius`:   vector of radii, length of the box in each dimension

Methods implemented:

    :(==), in #, etc ...

.
"""
struct Box{N,T <: AbstractFloat}
    center::SVector{N,T}
    radius::SVector{N,T}

    @propagate_inbounds function Box{N,T}(center, radius) where {N,T}

        @boundscheck begin
            if !( N == length(radius) == length(center) )
                throw(DimensionMismatch("Center vector and radius vector must have same length ($N)"))
            end

            if any(≤(0), radius)
                throw(DomainError(radius, "radius must be positive in every component"))
            end
        end

        return new{N,T}(SVector{N,T}(center), SVector{N,T}(radius))
    end
end

Box{T}(box::Box{N}) where {N,T} = Box{N,T}(box.center, box.radius)
Box{T}(args...) where {T} = Box{T}(Box(args...))
Box(c, r) = Box{length(c), F}(c, r)
Box(c::Number, r::Number) = Box((c,), (r,))

Box(int::IntervalBox{N}) where {N} = Box{N,F}(int)
Box(ints::Interval...) = Box(IntervalBox(ints...))
Box(ints::NTuple{N,<:Interval{T}}) where {N,T} = Box(ints...)

function Box{N,T}(intbox::IntervalBox{N}) where {N,T}
    ϵ = eps(T)
    c = ntuple(Val(N)) do i
        int = intbox.v[i]
        (int.hi .+ int.lo) ./ 2 .- ϵ
    end
    r = ntuple(Val(N)) do i
        int = intbox.v[i]
        (int.hi .- int.lo) ./ 2 .- ϵ
    end
    all(>(0), r) || return nothing
    @inbounds Box{N,T}(c, r)
end

function IntervalArithmetic.IntervalBox(box::Box{N,T}) where {N,T}
    c, r = box
    ϵ = eps(T)
    c, r = c .+ ϵ, r .+ ϵ
    IntervalBox{N,T}(c .± r)
end

function Base.show(io::IO, box::B) where {N,T,B<:Box{N,T}}
    c, r = box
    lo, hi = c - r, c + r
    if N ≤ 5
        join(io, ("[$l, $h)" for (l, h) in zip(lo, hi)), " × ")
    else
        print(io, "[$(lo[1]), $(hi[1])) × ... × [$(lo[end]), $(hi[end]))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", box::B) where {N,T,B<:Box{N,T}}
    c, r = box
    println(io, "$B: ")
    if N ≤ 5
        println(io, "   center: [$(join(c, ", "))]")
        println(io, "   radius: [$(join(r, ", "))]")
    else
        println(io, "   center: [$(c[1]), $(c[2]), ..., $(c[N-1]), $(c[N])]")
        println(io, "   radius: [$(r[1]), $(r[2]), ..., $(r[N-1]), $(r[N])]")
    end
end

Base.@propagate_inbounds function Base.in(point, box::Box)
    c, r = box
    @boundscheck begin
        M, N = length(point), ndims(box)
        if M != N
            throw(DimensionMismatch("point has dimension $M but box has dimension $N"))
        end
    end
    all((c - r) .<= point .< (c + r))
end

function Base.intersect(b1::Box{N}, b2::Box{N}) where {N}
    lo = max.(b1.center .- b1.radius, b2.center .- b2.radius)
    hi = min.(b1.center .+ b1.radius, b2.center .+ b2.radius)
    all(lo .< hi) || return nothing
    return Box((hi .+ lo) ./ 2, (hi .- lo) ./ 2)
end

Base.intersect(b1::IntervalBox, b2::Box) = Box(b1) ∩ b2
Base.intersect(b1::NTuple{N,<:Interval}, b2::Box{N}) where {N} = Box(b1) ∩ b2
Base.intersect(b1::Box, b2::IntervalBox) = b1 ∩ Box(b2)
Base.intersect(b1::Box{N}, b2::NTuple{N,<:Interval}) where {N} = b1 ∩ Box(b2)

Base.:(==)(b1::Box, b2::Box) = b1.center == b2.center && b1.radius == b2.radius
Base.length(::Box{N}) where {N} = N
Base.ndims(::Box{N}) where {N} = N
Base.eltype(::Type{Box{N,T}}) where {N,T} = T

Base.iterate(b::Box, i...) = (b.center, Val(:radius))
Base.iterate(b::Box, ::Val{:radius}) = (b.radius, Val(:done))
Base.iterate(b::Box, ::Val{:done}) = nothing

IntervalArithmetic.mince(b::Box{N,T}, n::Int) where {N,T} = mince(IntervalBox(b), n)
IntervalArithmetic.mince(b::Box{N,T}, ncuts::NTuple{N,Int}) where {N,T} = mince(IntervalBox(b), ncuts)

"""
    volume(box::Box)

Compute the volume of a box. 
"""
volume(box::Box) = prod(2 .* box.radius)
volume(::Nothing) = 0

"""
    vertices(box)

Return an iterator over the vertices of a `box = Box(center, radius)`. 
"""
function vertices(center::SVNT{N,T}, radius::SVNT{N,R}) where {N,T,R}
    I = CartesianIndices(ntuple(_->-1:2:1, N))
    (@muladd(center .+ radius .* Tuple(i)) for i in I)
end
vertices(box::Box) = vertices(box.center, box.radius)

function subdivide(box::Box{N,T}, dim) where {N,T} 
    c, r = box
    b1 = Box{N,T}(
        setindex(c, c[dim] - r[dim] / 2, dim),
        setindex(r, r[dim] / 2, dim)
    )
    b2 = Box{N,T}(
        setindex(c, c[dim] + r[dim] / 2, dim),
        setindex(r, r[dim] / 2, dim)
    )
    b1, b2
end

"""
    center(b::Box)

Return the center of a box. 
"""
center(box::Box) = box.center

"""
    radius(b::Box)

Return the radius of a box. 
"""
radius(box::Box) = box.radius

"""
    center(center, radius)

Return the center of a box as an iterable. 
Default function for `image_points` in `SampledBoxMap`s. 
"""
center(center, radius) = (center,)

"""
    rescale(box, point::Union{<:StaticVector{N,T}, <:NTuple{N,T}})
    rescale(center, radius, point::Union{<:StaticVector{N,T}, <:NTuple{N,T}})

Scale a `point` within the unit box ``[-1, 1]^N`` 
to lie within `box = Box(center, radius)`. 
"""
rescale(center, radius, point::SVNT{N,T}) where {N,T} = @muladd center .+ point .* radius
rescale(box::Box, points) = rescale(box.center, box.radius, points)

"""
    rescale(center, radius, points)

Return an iterable which calls
`rescale(center, radius, point)` for each point in `points`. 
"""
rescale(center, radius, points) = (rescale(center, radius, point) for point in points)

"""
    rescale(points)

Return a function 
```julia
(center, radius) -> rescale(center, radius, points)
```
Used in `domain_points` for `BoxMap`, `PointDiscretizedMap`. 
"""
rescale(points) = (center, radius) -> rescale(center, radius, points)
