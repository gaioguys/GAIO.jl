"""
    RegularPartition{N,T} <: BoxPartition{Box{N,T}}

A partition which divides a box into equally many parts on each axis,
depending on the `depth` parameter.

This allows a more compact representation than a `TreePartition`,
since we only need to store the maximum depth we are currently at.
It is comparable to a complete binary tree in the `TreePartition` sense.
"""
struct RegularPartition{N,T} <: BoxPartition{Box{N,T}}
    domain::Box{N,T}
    depth::Int
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

"""
    RegularPartition(domain::Box{N,T}, depth::Int)

Create a new `RegularPartition` on the given `domain` (radius must be non-negative)
and the given initial `depth`.
"""
function RegularPartition(domain::Box{N,T}, depth::Int) where {N,T}
    if any(x -> x <= 0, domain.radius)
        error("domain radius must be positive in every component")
    end

    left = domain.center .- domain.radius

    sd = map(let de = depth, dim = N
        i -> div(de - i, dim, RoundDown) + 1
    end, StaticArrays.SUnitRange(1, N))

    dims = 1 .<< sd

    scale = dims ./ (2 .* domain.radius)

    dimsprod_ = [SVector(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return RegularPartition(domain, depth, left, scale, dims, dimsprod)
end

"""
    RegularPartition(domain::Box)

Create a new `RegularPartition` on the given `domain` (radius must be non-negative).
"""
RegularPartition(domain) = RegularPartition(domain, 0)

"""
    dimension(partition::RegularPartition{N,T})

Return the dimension `N` of `partition`.
"""
dimension(partition::RegularPartition{N,T}) where {N,T} = N

"""
    depth(partition::RegularPartition)

Return the current partitioning depth of `partition`.
"""
depth(partition::RegularPartition) = partition.depth

"""
    subdivide(partition::RegularPartition)

Subdivide `partition`, effectively increasing the partitioning depth by 1.
"""
subdivide(partition::RegularPartition) = RegularPartition(partition.domain, partition.depth + 1)

keytype(::Type{<:RegularPartition}) = Int
keys_all(partition::RegularPartition) = 1:2^depth(partition)

Base.size(partition::RegularPartition) = partition.dims.data

"""
    key_to_box(partition::RegularPartition, key)

Return the box associated to some index `key` in the BoxPartition `partition`.
"""
function key_to_box(partition::RegularPartition{N,T}, key::Int) where {N,T}
    dims = size(partition)

    radius = partition.domain.radius ./ dims

    left = partition.domain.center .- partition.domain.radius

    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)

    return Box(center, radius)
end

"""
    point_to_key(partition::RegularPartition, point)

Map `point` to the key of the box in `partition` which contains `point` or return `nothing`, if there is not such box.
"""
function point_to_key(partition::RegularPartition, point)
    x = (point .- partition.left) .* partition.scale

    if any(x .< zero(eltype(x)))
        return nothing
    end

    x_ints = map(xi -> Base.unsafe_trunc(Int, xi), x)

    if any(x_ints .>= partition.dims)
        return nothing
    end

    return sum(x_ints .* partition.dimsprod) + 1
end
