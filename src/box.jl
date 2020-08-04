struct Box{N,T}
    center::SVector{N,T}
    radius::SVector{N,T}

    function Box(center, radius)
        N = length(center)

        if length(radius) != N
            error("center and radius must have the same length")
        end

        if any(x -> x < 0, radius)
            error("radius must be nonnegative in every component")
        end

        T = promote_type(eltype(center), eltype(radius))

        return new{N,T}(SVector{N,T}(center), SVector{N,T}(radius))
    end
end

Base.in(point, box::Box) = all(point .>= box.center - box.radius) && all(point .< box.center + box.radius)
