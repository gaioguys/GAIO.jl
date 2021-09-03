struct BoxPartition{N,T} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,Float64}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

function BoxPartition(domain::Box{N,T}, dims::NTuple{N,Int}) where {N,T}
    dims = SVector{N,Int}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (2*domain.radius)
    dimsprod_ = [SVector(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return BoxPartition(domain, left, scale, dims, dimsprod)
end

function BoxPartition(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int64,N)...)
    BoxPartition(domain, dims)
end

BoxPartition(domain::Box{1,T}, dims::Int) where {T} = BoxPartition(domain, (dims,))

dimension(partition::BoxPartition{N,T}) where {N,T} = N

function subdivide(P::BoxPartition{N,T}, dim::Int) where {N,T}
    new_dims = ntuple(i -> P.dims[i]*(i==dim ? 2 : 1), N)
    BoxPartition(P.domain, new_dims)
end

keytype(::Type{<:BoxPartition}) = Int
keys_all(partition::BoxPartition) = 1:prod(partition.dims)

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
