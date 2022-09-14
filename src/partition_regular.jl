"""
    BoxPartition(domain::Box{N}, dims::NTuple{N,<:Integer} = ntuple(_->1, N))

Data structure to partition a box into a 
`dims[1] x dims[2] x ... dims[N]` equidistant grid. 

Fields:
* `domain`:       box defining the entire domain
* `left`:         leftmost / bottom edge of the domain
* `scale`:        1 / diameter of each box in the new partition (componentwise)
* `dims`:         tuple, number of boxes in each dimension
* `dimsprod`:     for indexing the partition. `BoxPartition` uses linear indices, i.e.
                  keys are counted up in first dimension first, 
                  then second dimension, etc... 

```julia
         1st dim →
          * — * — * — *
    2nd   | 1 | 2 | 3 |
    dim   * — * — * — *
    ↓     | 4 | 5 | 6 |
          * — * — * — *
```

Methods implemented:

    :(==), ndims, size, length, keys, keytype #, etc ...

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
    scale = dims ./ (I(2) .* domain.radius)
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

"""
    subdivide(P::BoxPartition, dim) -> BoxPartition
    subdivide(B::BoxSet, dim) -> BoxSet

Bisect every box in `boxset` along the axis `dim`, 
giving rise to a new partition of the domain, with 
double the amount of boxes. 
"""
function subdivide(P::BoxPartition{N,T,I}, dim) where {N,T,I}
    new_dims = ntuple(i -> P.dims[i] * I(i==dim ? 2 : 1), N)
    return BoxPartition(P.domain, new_dims)
end

"""
    key_to_ints(P::BoxPartition, key)

Convert an index (linear or cartesian) in a `BoxPartition` to 
cartesian indices.
"""
function key_to_ints(partition::BoxPartition{N,T,I}, key::IndexTypes{N}) where {N,T,I}
    return NTuple{N,I}(CartesianIndices(size(partition))[key].I)
end

"""
    ints_to_box(P::BoxPartition, x_ints)

Return the box associated with the cartesian index `x_ints` 
within a `BoxPartition`.
"""
function ints_to_box(partition::BoxPartition{N,T}, x_ints::SVNT{N,I}) where {N,T,I<:Integer}
    radius = partition.domain.radius ./ partition.dims
    left = partition.domain.center .- partition.domain.radius
    center = @muladd left .+ radius .+ (2 .* radius) .* (x_ints .- 1)
    return Box{N,T}(center, radius)
end 

"""
    key_to_box(P::BoxPartition, key)

Return the box associated with the index within a `BoxPartition`. 
"""
function key_to_box(partition::BoxPartition{N,T,I}, key::IndexTypes{N}) where {N,T,I}
    x_ints = key_to_ints(partition, key)
    box = ints_to_box(partition, x_ints)
    return box
end

"""
    unsafe_point_to_ints(P::BoxPartiton, point)

Find the cartesian index for the box within a `BoxPartition` 
containing a point.

!!! danger "bounds checking"
    `unsafe_point_to_ints` does not do any bounds checking. The returned 
    cartesian index will be out of bounds if the point does not lie in the 
    partition. 
"""
function unsafe_point_to_ints(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    xi = (point .- partition.left) .* partition.scale
    return unsafe_trunc.(I, xi)
end

"""
    ints_to_key(P::BoxPartition, x_ints)

Convert linear to cartesian indices for a `BoxPartition`
"""
function ints_to_key(partition::BoxPartition{N,T,I}, x_ints) where {N,T,I<:Integer}
    if !all(zero(I) .<= x_ints .< size(partition))
        #@debug "point does not lie in the domain" point partition.domain
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + I(1)
    return key
end

"""
    point_to_key(P::BoxPartition, point)

Find the linear index for the box within a `BoxPartition` 
contatining a point. 

!!! note "Bounds checking"
    unlike `unsafe_point_to_ints`, `point_to_key` will return 
    `nothing` if the point does not lie in the partition. 
"""
function point_to_key(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    return key
end

"""
    point_to_box(P::BoxPartition, point)

Find the box within a `BoxPartition` containing a point. 
"""
function point_to_box(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    x_ints = unsafe_point_to_ints(partition, point)
    if !all(zero(I) .<= x_ints .< size(partition))
        return nothing
    end
    box = ints_to_box(partition, x_ints)
    return box
end