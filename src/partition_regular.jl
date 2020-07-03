struct RegularPartition{N,T} <: BoxPartition{Box{N,T}}
    domain::Box{N,T}
    depth::Int
end

RegularPartition(domain) = RegularPartition(domain, 0)

dimension(partition::RegularPartition{N,T}) where {N,T} = N
depth(partition::RegularPartition) = partition.depth
subdivide(partition::RegularPartition) = RegularPartition(partition.domain, partition.depth + 1)

keytype(::Type{<:RegularPartition}) = Int
keys_all(partition::RegularPartition) = 1:2^depth(partition)

function Base.size(partition::RegularPartition{N,T}) where {N,T}
    sd = map(let de = depth(partition), dim = dimension(partition)
        i -> div(de - i, dim, RoundDown) + 1
    end, StaticArrays.SUnitRange(1, N)).data

    return 1 .<< sd
end

function Base.getindex(partition::RegularPartition{N,T}, key::Int) where {N,T}
    dims = size(partition)

    radius = partition.domain.radius ./ dims

    left = partition.domain.center .- partition.domain.radius

    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)

    return Box(center, radius)
end

struct RegularPartitionKeys{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

function Base.keys(partition::RegularPartition{N,T}) where {N,T}
    left = partition.domain.center .- partition.domain.radius

    dims = SVector(size(partition))

    scale = dims ./ (2 .* partition.domain.radius)

    dimsprod_ = [SVector(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(dimension(partition))]

    return RegularPartitionKeys(left, scale, dims, dimsprod)
end

function Base.getindex(keys::RegularPartitionKeys, point)
    x = (point .- keys.left) .* keys.scale

    if any(x .< zero(eltype(x)))
        return nothing
    end

    x_ints = map(xi -> Base.unsafe_trunc(Int, xi), x)

    if any(x_ints .>= keys.dims)
        return nothing
    end

    return sum(x_ints .* keys.dimsprod) + 1
end

function subdivide_key(partition::RegularPartition, key::Int)
    box_indices = CartesianIndices(size(partition))

    sd = (depth(partition) % dimension(partition)) + 1

    linear_indices = LinearIndices(CartesianIndices(size(subdivide(partition))))

    box_index = box_indices[key].I

    child1 = Base.setindex(box_index, 2 * box_index[sd] - 1, sd)
    child2 = Base.setindex(box_index, 2 * box_index[sd], sd)

    return linear_indices[CartesianIndex(child1)], linear_indices[CartesianIndex(child2)]
end
