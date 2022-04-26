# abstract type AbstractBoxPartition{B <: Box} end

"""
A generalized box with
`center`:   vector where the box's center is located
`radius`:   vector of radii, length of the box in each dimension

"""
struct Box{N,T <: AbstractFloat}
    center::SVector{N,T}
    radius::SVector{N,T}

    function Box(center, radius)
        N = length(center)

        if length(radius) != N
            throw(DimensionMismatch("Center vector and radius vector must have same length ($N)"))
        end

        if any(x -> x <= 0, radius)
            throw(DomainError(radius, "radius must be positive in every component"))
        end
        
        T = promote_type(eltype(center), eltype(radius))
        if !(T <: AbstractFloat)
            T = Float64
        end

        return new{N,T}(SVector{N,T}(center), SVector{N,T}(radius))
    end
end

Base.in(point, box::Box) = all(box.center .- box.radius  .<=  point  .<  box.center .+ box.radius)

volume(box::Box) = prod(2 .* box.radius)

function Base.show(io::IO, box::Box) 
    print(io, "Box: center = $(box.center), radii = $(box.radius)")
end
