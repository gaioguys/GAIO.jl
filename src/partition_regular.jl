struct BoxPartition{N,T} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    depth::Int
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

function BoxPartition(domain::Box{N,T}; depth::Int=0) where {N,T}
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

    return BoxPartition(domain, depth, left, scale, dims, dimsprod)
end

# BoxPartition(domain) = BoxPartition(domain, depth = 0)

dimension(partition::BoxPartition{N,T}) where {N,T} = N
depth(partition::BoxPartition) = partition.depth
subdivide(partition::BoxPartition) = BoxPartition(partition.domain, depth=partition.depth + 1)

keytype(::Type{<:BoxPartition}) = Int
keys_all(partition::BoxPartition) = 1:2^depth(partition)

Base.size(partition::BoxPartition) = partition.dims.data

function Base.show(io::IO, partition::BoxPartition) 
    print(io, "$(partition.dims[1])")
    for k = 2:length(partition.left)
        print(io, " x $(partition.dims[k])")
    end 
    print(io, " BoxPartition")
end

function key_to_box(partition::BoxPartition{N,T}, key::Int) where {N,T}
    dims = size(partition)

    radius = partition.domain.radius ./ dims

    left = partition.domain.center .- partition.domain.radius

    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)

    return Box(center, radius)
end

function point_to_key(partition::BoxPartition, point)
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
