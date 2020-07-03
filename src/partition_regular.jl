struct RegularPartition{N,T} <: BoxPartition{Box{N,T}}
    domain::Box{N,T}
    depth::Int
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

function RegularPartition(domain::Box{N,T}, depth::Int) where {N,T}
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

RegularPartition(domain) = RegularPartition(domain, 0)

dimension(partition::RegularPartition{N,T}) where {N,T} = N
depth(partition::RegularPartition) = partition.depth
subdivide(partition::RegularPartition) = RegularPartition(partition.domain, partition.depth + 1)

keytype(::Type{<:RegularPartition}) = Int
keys_all(partition::RegularPartition) = 1:2^depth(partition)

Base.size(partition::RegularPartition) = partition.dims.data

function key_to_box(partition::RegularPartition{N,T}, key::Int) where {N,T}
    dims = size(partition)

    radius = partition.domain.radius ./ dims

    left = partition.domain.center .- partition.domain.radius

    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)

    return Box(center, radius)
end

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
