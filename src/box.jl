const F = (Int == Int64) ? Float64 : Float32

# abstract type AbstractBoxPartition{B <: Box} end
"""
A generalized box with
`center`:   vector where the box's center is located
`radius`:   vector of radii, length of the box in each dimension

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

Base.in(point, box::Box) = all(box.center .- box.radius  .<=  point  .<  box.center .+ box.radius)
Base.:(==)(b1::Box, b2::Box) = false
Base.:(==)(b1::Box{N}, b2::Box{N}) where N = (all(b1.center .== b2.center) && all(b1.radius .== b2.radius))

volume(box::Box) = prod(2 .* box.radius)

function Base.show(io::IO, box::Box) 
    print(io, "Box:\n   center = $(box.center),\n   radii = $(box.radius)")
end
