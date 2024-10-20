"""
    GridPartition(domain::Box{N}, dims::NTuple{N,<:Integer} = ntuple(_->1, N)) 

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
struct GridPartition{N,T,I<:Integer} <: BoxLayout{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
end

function GridPartition(domain::Box{N,T}, dims::SVNT{N,I}) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (2 .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    return GridPartition{N,T,I}(domain, left, scale, dims)
end

function GridPartition{I}(domain::Box{N,T}) where {N,T,I}
    dims = ntuple(_->one(I), Val(N))
    GridPartition(domain, dims)
end

GridPartition(domain::Box{N,T}) where {N,T} = GridPartition{Int}(domain)

function GridPartition(domain::Box{1}, dims::Integer)
    GridPartition(domain, (dims,))
end

Base.:(==)(p1::GridPartition, p2::GridPartition) = p1.domain == p2.domain && p1.dims == p2.dims
Base.ndims(::GridPartition{N}) where {N} = N
Base.size(partition::GridPartition) = partition.dims.data # .data returns as tuple
Base.length(partition::GridPartition) = prod(partition.dims)
Base.keytype(::Type{<:GridPartition{N,T,I}}) where {N,T,I} = NTuple{N,I}

center(partition::GridPartition) = center(partition.domain)
radius(partition::GridPartition) = radius(partition.domain)

Base.CartesianIndices(partition::GridPartition) = CartesianIndices(size(partition))
Base.LinearIndices(partition::GridPartition) = LinearIndices(size(partition))

Base.checkbounds(::Type{Bool}, partition::GridPartition{N,T,I}, key) where {N,T,I} = all(1 .≤ key .≤ size(partition))
Base.checkbounds(::Type{Bool}, partition::GridPartition{N,T,I}, ::Nothing) where {N,T,I} = false

function Base.keys(partition::P) where {P<:GridPartition}
    K = keytype(P)
    (K(i.I) for i in CartesianIndices(partition))
end

function Base.show(io::IO, partition::P) where {N,P<:GridPartition{N}}
    if N ≤ 5
        print(io, join(size(partition), " x "), " - element GridPartition")
    else
        sz = size(partition)
        print(io, sz[1], " x ", sz[2], " ... ", sz[N-1], " x ", sz[N], " - element GridPartition")
    end
end

"""
    subdivide(P::GridPartition, dim) -> GridPartition
    subdivide(B::BoxSet, dim) -> BoxSet

Bisect every box in the `GridPartition` or `BoxSet` 
along the axis `dim`, giving rise to a new partition 
of the domain, with double the amount of boxes. 
"""
function subdivide(P::GridPartition{N,T,I}, dim) where {N,T,I}
    new_dims = setindex(P.dims, 2 * P.dims[dim], dim)
    new_scale = setindex(P.scale, 2 * P.scale[dim], dim)
    return GridPartition{N,T,I}(P.domain, P.left, new_scale, new_dims)
end

subdivide(P::GridPartition{N,T,I}) where {N,T,I} = subdivide(P, argmin(P.dims))

"""
    marginal(P::GridPartition{N}; dim) -> GridPartition{N-1}

Construct the projection of a `GridPartition` along an axis given by 
its dimension `dim`. 
"""
function marginal(P⁺::GridPartition; dim)
    cen⁺, rad⁺ = P⁺.domain
    dims⁺ = size(P⁺)

    cen = tuple_deleteat(Tuple(cen⁺), dim)
    rad = tuple_deleteat(Tuple(rad⁺), dim)
    dims = tuple_deleteat(dims⁺, dim)

    return GridPartition( Box(cen, rad), dims )
end

"""
    key_to_box(P::GridPartition, key)

Return the box associated with the index 
within a `GridPartition`. 
"""
@propagate_inbounds function key_to_box(partition::GridPartition{N,T,I}, x_ints) where {N,T,I}
    @boundscheck checkbounds(Bool, partition, x_ints) || throw(BoundsError(partition, x_ints))
    radius = partition.domain.radius ./ partition.dims
    left = partition.left
    center = @muladd @. left + radius + (2 * radius) * (x_ints - 1)
    return Box{N,T}(center, radius)
end

key_to_box(partition::GridPartition{N,T,I}, x_ints::CartesianIndex) where {N,T,I} = key_to_box(partition, x_ints.I)
key_to_box(partition::GridPartition{N,T,I}, x_ints::Nothing) where {N,T,I} = nothing

"""
    point_to_key(P::GridPartition, point)

Find the index for the box within a `GridPartition` 
contatining a point, or `nothing` if the point does 
not lie in the domain. 
"""
@propagate_inbounds function point_to_key(partition::GridPartition{N,T,I}, point) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, partition, x_ints)
        @debug "something went wrong in point_to_key" point xi x_ints partition partition.domain
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(partition))
    end
    return x_ints
end

"""
    bounded_point_to_key(P::GridPartition, point)

Find the cartesian index of the nearest box within a 
`GridPartition` to a point. Conicides with `point_to_key` 
if the point lies in the partition. Default behavior 
is to set `NaN = Inf` if `NaN`s are present in `point`. 
"""
function bounded_point_to_key(partition::BoxLayout{B}, point) where {N,T,B<:Box{N,T}}
    center, radius = partition.domain
    small_bound = 1 ./ ( 2 .* partition.scale )
    left = center .- radius .+ small_bound
    right = center .+ radius .- small_bound
    p = ifelse.(isnan.(point), convert(T, Inf), point)
    p = min.(max.(p, left), right)
    return point_to_key(partition, p)
end

"""
    point_to_box(P::BoxLayout, point)

Find the box within a `GridPartition` containing a point. 
"""
function point_to_box(partition::BoxLayout, point)
    x_ints = point_to_key(partition, point)
    isnothing(x_ints) && return x_ints
    return @inbounds key_to_box(partition, x_ints)
end
