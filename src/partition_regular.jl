"""
    BoxPartition(
        domain::Box{N}, 
        dims::NTuple{N,<:Integer} = ntuple(_->1, N)
        ; indextype = (N < 9 : IndexLinear() : IndexCartesian())
    ) 

Data structure to partition a domain into a 
`dims[1] x dims[2] x ... dims[N]` equidistant box grid. 

Fields:
* `domain`:         box defining the entire domain
* `left`:           leftmost / bottom edge of the domain
* `scale`:          1 / diameter of each box in the new partition (componentwise)
* `dims`:           tuple, number of boxes in each dimension
* `dimsprod`:       for indexing the partition. `BoxPartition` uses linear indices, i.e.
                    keys are counted up in first dimension first, 
                    then second dimension, etc... 
* `indextype`:      whether the partition will use cartesian or 
                    linear indices by default. Values can be 
                    `IndexLinear()`, `IndexCartesian()`

Methods implemented:

    :(==), ndims, size, length, keys, keytype #, etc ...

.
"""
struct BoxPartition{N,T,I<:Integer,A<:Union{IndexCartesian,IndexLinear}} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
    dimsprod::SVector{N,I}
    indextype::A
end

function BoxPartition(domain::Box{N,T}, dims::NTuple{N,I}; indextype=(N < 10 ? IndexLinear() : IndexCartesian())) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (I(2) .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    dimsprod_ = [SVector{1,I}(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return BoxPartition{N,T,I,typeof(indextype)}(domain, left, scale, dims, dimsprod, indextype)
end

function BoxPartition(domain::Box{N,T}; indextype=(N < 10 ? IndexLinear() : IndexCartesian())) where {N,T}
    dims = tuple(ones(Int,N)...)
    BoxPartition(domain, dims; indextype=indextype)
end

function BoxPartition(domain::Box{1}, dims::Integer; indextype=IndexLinear())
    BoxPartition(domain, (dims,); indextype=indextype)
end

Base.:(==)(p1::BoxPartition, p2::BoxPartition) = p1.domain == p2.domain && p1.dims == p2.dims
Base.ndims(::BoxPartition{N}) where {N} = N
Base.size(partition::BoxPartition) = partition.dims.data # .data returns as tuple
Base.length(partition::BoxPartition) = partition.dimsprod[end] * partition.dims[end] # == prod(partition.dims)
Base.keytype(::Type{<:BoxPartition{N,T,I,L}}) where {N,T,I,L<:IndexLinear} = I
Base.keytype(::Type{<:BoxPartition{N,T,I,L}}) where {N,T,I,L<:IndexCartesian} = NTuple{N,I}
Base.CartesianIndices(partition::BoxPartition) = CartesianIndices(size(partition))
Base.LinearIndices(partition::BoxPartition) = LinearIndices(size(partition))
Base.keys(partition::BoxPartition{N,T,I,IndexLinear}) where {N,T,I} = one(I) : length(partition)
Base.keys(partition::BoxPartition{N,T,I,IndexCartesian}) where {N,T,I} = (NTuple{N,I}(i) for i in CartesianIndices(partition))

function Base.show(io::IO, partition::P) where {P<:BoxPartition}
    print(io, join(size(partition), " x "), " - element $P")
end

"""
    subdivide(P::BoxPartition, dim) -> BoxPartition
    subdivide(B::BoxSet, dim) -> BoxSet

Bisect every box in `boxset` along the axis `dim`, 
giving rise to a new partition of the domain, with 
double the amount of boxes. 
"""
function subdivide(P::BoxPartition{N,T,I}, dim) where {N,T,I}
    new_dims = ntuple(i -> P.dims[i] * I(i==dim ? 2 : 1), N)
    return BoxPartition(P.domain, new_dims; indextype=P.indextype)
end

function linear_to_cartesian(partition::BoxPartition{N,T,I}, key) where {N,T,I}
    return NTuple{N,I}(CartesianIndices(partition)[key].I)
end

function cartesian_to_box(partition::BoxPartition{N,T}, x_ints) where {N,T}
    radius = partition.domain.radius ./ partition.dims
    left = partition.domain.center .- partition.domain.radius
    center = @muladd left .+ radius .+ (2 .* radius) .* (x_ints .- 1)
    return Box{N,T}(center, radius)
end 

cartesian_to_key(partition::BoxPartition{N,T,I,IndexCartesian}, x_ints) where {N,T,I} = x_ints
cartesian_to_key(partition::BoxPartition{N,T,I,IndexLinear}, x_ints) where {N,T,I} = cartesian_to_linear(partition, x_ints)

"""
    key_to_box(P::BoxPartition, key)

Return the box associated with the index 
within a `BoxPartition`. 
"""
key_to_box(partition::BoxPartition{N,T,I,IndexCartesian}, key) where {N,T,I} = cartesian_to_box(partition, key)

function key_to_box(partition::BoxPartition{N,T,I,IndexLinear}, key) where {N,T,I}
    x_ints = linear_to_cartesian(partition, key)
    return cartesian_to_box(partition, x_ints)
end

"""
    unsafe_point_to_cartesian(P::BoxPartiton, point)

Find the cartesian index for the box within a 
`BoxPartition` containing a point.

!!! danger "bounds checking"
    `unsafe_point_to_ints` does not do any bounds checking. The returned 
    cartesian index will be out of bounds if the point does not lie in the 
    partition. 
"""
function unsafe_point_to_cartesian(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    xi = (point .- partition.left) .* partition.scale
    return ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
end

"""
    bounded_point_to_cartesian(partition::BoxPartition, point)

Find the cartesian index of the nearest box within a 
`BoxPartition` to a point. Conicides with `unsafe_point_to_ints` 
if the point lies in the partition. 
"""
function bounded_point_to_cartesian(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    xi = (point .- partition.left) .* partition.scale
    xi = max.(zero(I), xi)
    xi = min.(size(partition) .- one(I), xi)
    return ntuple( i -> trunc(I, xi[i]) + one(I), Val(N) )
end

function cartesian_to_linear(partition::BoxPartition{N,T,I}, x_ints) where {N,T,I<:Integer}
    key = sum((x_ints .- one(I)) .* partition.dimsprod) + one(I)
    return key
end

"""
    point_to_key(P::BoxPartition, point)

Find the index for the box within a `BoxPartition` 
contatining a point. 

!!! note "Bounds checking"
    unlike `unsafe_point_to_ints`, `point_to_key` will return 
    `nothing` if the point does not lie in the partition. 
"""
function point_to_key(partition::BoxPartition{N,T,I,IndexCartesian}, point) where {N,T,I}
    point in partition.domain || return nothing
    x_ints = unsafe_point_to_cartesian(partition, point)
    return x_ints
end

function point_to_key(partition::BoxPartition{N,T,I,IndexLinear}, point) where {N,T,I}
    point in partition.domain || return nothing
    x_ints = unsafe_point_to_cartesian(partition, point)
    return cartesian_to_linear(partition, x_ints)
end

"""
    point_to_box(P::BoxPartition, point)

Find the box within a `BoxPartition` containing a point. 
"""
function point_to_box(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    point in partition.domain || return nothing
    x_ints = unsafe_point_to_cartesian(partition, point)
    return cartesian_to_box(partition, x_ints)
end