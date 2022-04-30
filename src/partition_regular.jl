"""
Internal data structure to partition a box

`domain`:       box defining the entire domain

`left`:         leftmost / bottom edge of the domain

`scale`:        1 / diameter of each box in the new partition (componentwise)

`dimsprod`:     for indexing the partition:
                key is counted up in first dimension first, then second dimension, etc... ie

                 1st dim →
                  * — * — * — *
            2nd   | 1 | 2 | 3 |
            dim   * — * — * — *
            ↓     | 4 | 5 | 6 |
                  * — * — * — *
.
"""
struct BoxPartition{N,T,I} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
    dimsprod::SVector{N,I}
end

function BoxPartition(domain::Box{N,T}, dims::NTuple{N,I}) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (2 .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    dimsprod_ = [SVector(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return BoxPartition(domain, left, scale, dims, dimsprod)
end

function BoxPartition(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int32,N)...)
    BoxPartition(domain, dims)
end

BoxPartition(domain::Box{1,T}, dims::I) where {T,I} = BoxPartition(domain, (dims,))

dimension(::BoxPartition{N,T,I}) where {N,T,I} = N

function subdivide(P::BoxPartition{N,T}, dim::I) where {N,T,I}
    new_dims = ntuple(i -> P.dims[i]*(i==dim ? 2 : 1), N)
    return BoxPartition(P.domain, new_dims)
end

keytype(::Type{<:BoxPartition{N,T,I}}) where {N,T,I} = I
keys_all(partition::BoxPartition) = 1:prod(partition.dims)
# == 1 : partition.dimsprod[end] * partition.dims[end]

Base.size(partition::BoxPartition) = partition.dims.data # .data returns as tuple

function Base.show(io::IO, partition::BoxPartition) 
    print(io, "$(partition.dims[1])")
    for k = 2:length(partition.left)
        print(io, " x $(partition.dims[k])")
    end 
    print(io, " BoxPartition")
end

@muladd @propagate_inbounds function key_to_box(
        partition::BoxPartition{N,T,I}, key::M
    ) where M <: Union{<:Integer, NTuple{N, <:Integer}} where {N,T,I}

    dims = size(partition)
    radius = partition.domain.radius ./ dims
    left = partition.domain.center .- partition.domain.radius
    center = @. left + radius + 2 * radius * ($(CartesianIndices(dims)[key].I) - 1)
    # start at leftmost box in the partition and move $key boxes right
    return Box(center, radius)
end 

@propagate_inbounds function unsafe_point_to_ints(partition::BoxPartition, point)
    x = (point .- partition.left) .* partition.scale    
    # counts how many boxes x is away from left (componentwise)
    return map(xi -> unsafe_trunc(Int, xi), x)
end

@propagate_inbounds function ints_to_key(partition::BoxPartition, x_ints)
    if any(x_ints .< zero(eltype(x_ints))) || any(x_ints .>= partition.dims)
        #@debug "point does not lie in the domain" point partition.domain
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + 1
    return key
end

@propagate_inbounds function point_to_key(partition::BoxPartition, point)
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    return key
end
