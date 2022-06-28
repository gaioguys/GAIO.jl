"""
Internal data structure to partition a box

`domain`:       box defining the entire domain

`left`:         leftmost / bottom edge of the domain

`scale`:        1 / diameter of each box in the new partition (componentwise)

`dims`:         tuple, number of boxes in each dimension

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
struct BoxPartition{N,T,I<:Integer} <: AbstractBoxPartition{Box{N,T}}
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
    dimsprod_ = [SVector{1,I}(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return BoxPartition{N,T,I}(domain, left, scale, dims, dimsprod)
end

function BoxPartition(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int,N)...)
    BoxPartition(domain, dims)
end

BoxPartition(domain::Box{1}, dims::Integer) = BoxPartition(domain, (dims,))

Base.:(==)(p1::BoxPartition, p2::BoxPartition) = p1.domain == p2.domain && p1.dims == p2.dims
Base.ndims(::BoxPartition{N}) where {N} = N
Base.size(partition::BoxPartition) = partition.dims.data # .data returns as tuple
Base.length(partition::BoxPartition) = partition.dimsprod[end] * partition.dims[end] # == prod(partition.dims)
Base.keytype(::Type{<:BoxPartition{N,T,I}}) where {N,T,I} = I
Base.keys(partition::BoxPartition) = 1 : length(partition)

function Base.show(io::IO, partition::BoxPartition) 
    print(io, join(size(partition), " x "), "  BoxPartition")
end

function subdivide(P::BoxPartition{N,T,I}, dim::Integer) where {N,T,I}
    new_dims = ntuple(i -> P.dims[i] * I(i==dim ? 2 : 1), N)
    return BoxPartition(P.domain, new_dims)
end

# TODO: replace with overloaded getindex
@muladd function key_to_box(
        partition::BoxPartition{N,T}, key::M
    ) where {N,T,M<:Union{<:Integer, NTuple{N,<:Integer}}}

    radius = partition.domain.radius ./ partition.dims
    left = partition.domain.center .- partition.domain.radius
    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(size(partition))[key].I .- 1)
    # start at leftmost box in the partition and move $key boxes right
    return Box{N,T}(center, radius)
end 

function unsafe_point_to_ints(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    xi = (point .- partition.left) .* partition.scale    
    # counts how many boxes x is away from left (componentwise)
    return unsafe_trunc.(I, xi)
end

function ints_to_key(partition::BoxPartition{N,T,I}, x_ints) where {N,T,I}
    if !all(zero(I) .<= x_ints .< size(partition))
        #@debug "point does not lie in the domain" point partition.domain
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + 1
    return key
end

function point_to_key(partition::BoxPartition, point)
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    return key
end
