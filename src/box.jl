struct Box{N,T}
    center::SVector{N,T}
    radius::SVector{N,T}

    function Box(center, radius)
        N = length(center)

        if length(radius) != N
            error("center and radius must have the same length")
        end

        T = promote_type(eltype(center), eltype(radius))

        return new{N,T}(SVector{N,T}(center), SVector{N,T}(radius))
    end
end
