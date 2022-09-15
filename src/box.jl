const F = (Int == Int64) ? Float64 : Float32

Core.@doc raw"""
    Box{N,T}(center, radius)
    Box(center, radius)

A generalized box in dimension `N` with element type `T`. 
Mathematically, this is a set
```math
[center_1 - radius_1,\ center_1 + radius_1) \ \times \ [center_N - radius_N,\ center_N + radius_N)
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

    function Box{N,T}(center, radius) where {N,T}

        if !( N == length(radius) == length(center) )
            throw(DimensionMismatch("Center vector and radius vector must have same length ($N)"))
        end

        if any(≤(0), radius)
            throw(DomainError(radius, "radius must be positive in every component"))
        end

        return new{N,T}(SVector{N,T}(center), SVector{N,T}(radius))
    end
end

Box(center, radius) = Box{length(center), promote_type(F, eltype(center), eltype(radius))}(center, radius)

function Base.show(io::IO, box::B) where {B<:Box} 
    print(io, "$(B):\n    center = $(box.center),\n    radii = $(box.radius)")
end

Base.in(point, box::Box) = all(box.center .- box.radius  .<=  point  .<  box.center .+ box.radius)
Base.:(==)(b1::Box, b2::Box) = b1.center == b2.center && b1.radius == b2.radius

"""
Computes the volume of a box. 
"""
volume(box::Box) = prod(2 .* box.radius)

"""
    vertices(box)
    vertices(center, radius)

Return an iterator over the vertices of a `box = Box(center, radius)`. 
"""
function vertices(center::SVNT{N,T}, radius::SVNT{N,T}) where {N,T}
    I = CartesianIndices(ntuple(_->-1:2:1, N))
    (@muladd(center .+ radius .* Tuple(i)) for i in I)
end
vertices(box::Box) = vertices(box.center, box.radius)
