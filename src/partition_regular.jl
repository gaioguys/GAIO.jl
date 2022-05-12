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
struct BoxPartition{N,T,I,D} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
    dimsprod::SVector{N,I}

    function BoxPartition(
            domain::Box{N,T}, left::SVector{N,T}, scale::SVector{N,T}, dims::SVector{N,I}, dimsprod::SVector{N,I}
        ) where {N,T,I}

        return new{N,T,I,dims.data}(domain, left, scale, dims, dimsprod)
    end
end

function BoxPartition(domain::Box{N,T}, dims::NTuple{N,I}) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (I(2) .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    dimsprod_ = [SVector(I(1)); cumprod(dims)]
    dimsprod = SVector{N,I}(dimsprod_[SOneTo(N)])

    return BoxPartition(domain, left, scale, dims, dimsprod)
end

function BoxPartition(domain::Box{N,T}, I=Int32) where {N,T}
    dims = tuple(ones(I,N)...)
    BoxPartition(domain, dims)
end

BoxPartition(domain::Box{1,T}, dims::I) where {T,I} = BoxPartition(domain, (dims,))

dimension(::BoxPartition{N,T,I}) where {N,T,I} = N

function subdivide(P::BoxPartition{N,T,I,D}, dim) where {N,T,I,D}
    new_dims = ntuple(i -> D[i]*(i==dim ? I(2) : I(1)), N)
    return BoxPartition(P.domain, new_dims)
end

Base.size(::BoxPartition{N,T,I,D}) where {N,T,I,D} = D
Base.length(::BoxPartition{N,T,I,D}) where {N,T,I,D} = prod(D)

keytype(::Type{<:BoxPartition{N,T,I}}) where {N,T,I} = I
keys_all(partition::BoxPartition) = 1:length(partition)
# == 1 : partition.dimsprod[end] * partition.dims[end]

function Base.show(io::IO, partition::BoxPartition{N,T,I,D}) where {N,T,I,D}
    print(io, join(D, " x "), " BoxPartition")
end

@muladd @propagate_inbounds function key_to_cr(
        partition::BoxPartition{N,T,I,D}, key::M
    ) where M <: Union{<:Integer, NTuple{N, <:Integer}} where {N,T,I,D}

    radius = partition.domain.radius ./ D
    left = partition.domain.center .- partition.domain.radius
    center = @. left + radius + 2 * radius * ($(CartesianIndices(D)[key].I) - 1)
    # start at leftmost box in the partition and move $key boxes right
    return center, radius
end 

@propagate_inbounds function key_to_box(
        partition::BoxPartition{N,T,I,D}, key::M
    ) where M <: Union{<:Integer, NTuple{N, <:Integer}} where {N,T,I,D}

    return Box(key_to_cr(partition, key)...)
end 

@propagate_inbounds function unsafe_point_to_ints(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    x = (point .- partition.left) .* partition.scale    
    # counts how many boxes x is away from left (componentwise)
    return unsafe_trunc.(I, x)
end

@propagate_inbounds function ints_to_key(partition::BoxPartition{N,T,I,D}, x_ints) where {N,T,I,D}
    if any(x_ints .< zero(I)) || any(x_ints .>= D)
        #@debug "point does not lie in the domain" point partition.domain
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + I(1)
    return key
end

@propagate_inbounds function point_to_key(partition::BoxPartition, point)
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    return key
end
