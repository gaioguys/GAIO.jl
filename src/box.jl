"""
    Box{N, T}

A box in dimension `N` centered around `center` with size `2 .* radius`, both of which are (stack-allocated) vectors with a value type `T`.
It is represented by a cartesian product of half-intervals which are closed on the lower end, i.e. the box is of the form:

[center[1] - radius[1], center[1] + radius[1]) × ... × [center[N] - radius[N], center[N] + radius[N])
"""
struct Box{N,T}
    center::SVector{N,T}
    radius::SVector{N,T}

    """
    Box(center, radius)

    Create a new box with given `center` and `radius`.
    The `radius` should be non-negative in all dimensions/components,
    and the dimensions of `center` and `radius` should match.
    """
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

"""
    in(point, box::Box)

Check if some point lies inside the given `box`.
It includes the boundary on the lesser side in each dimension, but not the boundary on the larger side.

See also: the documentation for the type `Box`.
"""
Base.in(point, box::Box) = all(point .>= box.center - box.radius) && all(point .< box.center + box.radius)
