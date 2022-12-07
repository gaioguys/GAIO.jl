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

Methods implemented:

    :(==), ndims, size, length, keys, keytype #, etc ...

.
"""
struct BoxPartition{N,T,I<:Integer} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
end

function BoxPartition(domain::Box{N,T}, dims::SVNT{N,I}) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (I(2) .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box

    return BoxPartition{N,T,I}(domain, left, scale, dims)
end

function BoxPartition(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int,N)...)
    BoxPartition(domain, dims)
end

function BoxPartition(domain::Box{1}, dims::Integer)
    BoxPartition(domain, (dims,))
end

Base.:(==)(p1::BoxPartition, p2::BoxPartition) = p1.domain == p2.domain && p1.dims == p2.dims
Base.ndims(::BoxPartition{N}) where {N} = N
Base.size(partition::BoxPartition) = partition.dims.data # .data returns as tuple
Base.length(partition::BoxPartition) = prod(partition.dims)
Base.keytype(::Type{<:BoxPartition{N,T,I}}) where {N,T,I} = NTuple{N,I}

function Base.keys(partition::P) where {P<:BoxPartition}
    K = keytype(P)
    (K(i.I) for i in CartesianIndices(partition))
end

Base.CartesianIndices(partition::BoxPartition) = CartesianIndices(size(partition))
Base.LinearIndices(partition::BoxPartition) = LinearIndices(size(partition))

Base.checkbounds(::Type{Bool}, partition::BoxPartition{N,T,I}, key) where {N,T,I} = all(i -> 1 ≤ key[i] ≤ size(partition)[i], 1:N)

function Base.show(io::IO, partition::P) where {N,P<:BoxPartition{N}}
    if N ≤ 5
        print(io, join(size(partition), " x "), " - element $P")
    else
        sz = size(partition)
        print(io, sz[1], " x ", sz[2], " ... ", sz[N-1], " x ", sz[N], " - element $P")
    end
end

function Base.show(io::IO, ::MIME"text/plain", partition::P) where {N,P<:BoxPartition{N}}
    if N ≤ 5
        print(io, join(size(partition), " x "), " - element BoxPartition")
    else
        sz = size(partition)
        print(io, sz[1], " x ", sz[2], " ... ", sz[N-1], " x ", sz[N], " - element BoxPartition")
    end
end

"""
    subdivide(P::BoxPartition, dim) -> BoxPartition
    subdivide(B::BoxSet, dim) -> BoxSet

Bisect every box in the `BoxPartition` or `BoxSet` 
along the axis `dim`, giving rise to a new partition 
of the domain, with double the amount of boxes. 
"""
function subdivide(P::BoxPartition{N,T,I}, dim) where {N,T,I}
    new_dims = setindex(P.dims, P.dims[dim] * 2, dim)
    return BoxPartition(P.domain, new_dims)
end

"""
    key_to_box(P::BoxPartition, key)

Return the box associated with the index 
within a `BoxPartition`. 
"""
function key_to_box(partition::BoxPartition{N,T}, x_ints) where {N,T}
    radius = partition.domain.radius ./ partition.dims
    left = partition.left
    center = @muladd @. left + radius + (2 * radius) * (x_ints - 1)
    return Box{N,T}(center, radius)
end

"""
    point_to_key(P::BoxPartition, point)

Find the index for the box within a `BoxPartition` 
contatining a point, or `nothing` if the point does 
not lie in the domain. 
"""
function point_to_key(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    return ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
end

"""
    bounded_point_to_key(partition::BoxPartition, point)

Find the cartesian index of the nearest box within a 
`BoxPartition` to a point. Conicides with `point_to_key` 
if the point lies in the partition. 
"""
function bounded_point_to_key(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    xi = (point .- partition.left) .* partition.scale
    xi = max.(zero(I), xi)
    xi = min.(size(partition) .- one(I), xi)
    return ntuple( i -> trunc(I, xi[i]) + one(I), Val(N) )
end

"""
    point_to_box(P::BoxPartition, point)

Find the box within a `BoxPartition` containing a point. 
"""
function point_to_box(partition::BoxPartition{N,T,I}, point) where {N,T,I}
    point in partition.domain || return nothing
    x_ints = unsafe_point_to_key(partition, point)
    return key_to_box(partition, x_ints)
end
