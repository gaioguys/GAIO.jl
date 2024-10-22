"""
    BoxGrid(domain::Box{N}, dims::NTuple{N,<:Integer} = ntuple(_->1, N)) 

Data structure to grid a domain into a 
`dims[1] x dims[2] x ... dims[N]` equidistant box grid. 

Fields:
* `domain`:         box defining the entire domain
* `left`:           leftmost / bottom edge of the domain
* `scale`:          1 / diameter of each box in the new grid (componentwise)
* `dims`:           tuple, number of boxes in each dimension

Methods implemented:

    :(==), ndims, size, length, keys, keytype #, etc ...

.
"""
struct BoxGrid{N,T,I<:Integer} <: BoxLayout{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,I}
end

function BoxGrid(domain::Box{N,T}, dims::SVNT{N,I}) where {N,T,I}
    dims = SVector{N,I}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (2 .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    return BoxGrid{N,T,I}(domain, left, scale, dims)
end

function BoxGrid{I}(domain::Box{N,T}) where {N,T,I}
    dims = ntuple(_->one(I), Val(N))
    BoxGrid(domain, dims)
end

BoxGrid(domain::Box{N,T}) where {N,T} = BoxGrid{Int}(domain)

function BoxGrid(domain::Box{1}, dims::Integer)
    BoxGrid(domain, (dims,))
end

Base.:(==)(p1::BoxGrid, p2::BoxGrid) = p1.domain == p2.domain && p1.dims == p2.dims
Base.ndims(::BoxGrid{N}) where {N} = N
Base.size(G::BoxGrid) = G.dims.data # .data returns as tuple
Base.length(G::BoxGrid) = prod(G.dims)
Base.keytype(::Type{<:BoxGrid{N,T,I}}) where {N,T,I} = NTuple{N,I}

center(G::BoxGrid) = center(G.domain)
radius(G::BoxGrid) = radius(G.domain)

Base.CartesianIndices(G::BoxGrid) = CartesianIndices(size(G))
Base.LinearIndices(G::BoxGrid) = LinearIndices(size(G))

Base.checkbounds(::Type{Bool}, G::BoxGrid{N,T,I}, key) where {N,T,I} = all(1 .≤ key .≤ size(G))
Base.checkbounds(::Type{Bool}, G::BoxGrid{N,T,I}, ::Nothing) where {N,T,I} = false

function Base.keys(G::P) where {P<:BoxGrid}
    K = keytype(P)
    (K(i.I) for i in CartesianIndices(G))
end

function Base.show(io::IO, G::P) where {N,P<:BoxGrid{N}}
    if N ≤ 5
        print(io, join(size(G), " x "), " - element BoxGrid")
    else
        sz = size(G)
        print(io, sz[1], " x ", sz[2], " ... ", sz[N-1], " x ", sz[N], " - element BoxGrid")
    end
end

"""
    subdivide(P::BoxGrid, dim) -> BoxGrid
    subdivide(B::BoxSet, dim) -> BoxSet

Bisect every box in the `BoxGrid` or `BoxSet` 
along the axis `dim`, giving rise to a new grid 
of the domain, with double the amount of boxes. 
"""
function subdivide(G::BoxGrid{N,T,I}, dim) where {N,T,I}
    new_dims = setindex(G.dims, 2 * G.dims[dim], dim)
    new_scale = setindex(G.scale, 2 * G.scale[dim], dim)
    return BoxGrid{N,T,I}(G.domain, G.left, new_scale, new_dims)
end

subdivide(G::BoxGrid{N,T,I}) where {N,T,I} = subdivide(G, argmin(G.dims))

"""
    marginal(G::BoxGrid{N}; dim) -> BoxGrid{N-1}

Construct the projection of a `BoxGrid` along an axis given by 
its dimension `dim`. 
"""
function marginal(G⁺::BoxGrid; dim)
    center⁺, radius⁺ = G⁺.domain
    dims⁺ = size(G⁺)

    center = tuple_deleteat(Tuple(center⁺), dim)
    radius = tuple_deleteat(Tuple(radius⁺), dim)
    dims = tuple_deleteat(dims⁺, dim)

    return BoxGrid( Box(center, radius), dims)
end

"""
    key_to_box(G::BoxGrid, key)

Return the box associated with the index 
within a `BoxGrid`. 
"""
@propagate_inbounds function key_to_box(G::BoxGrid{N,T,I}, x_ints) where {N,T,I}
    @boundscheck checkbounds(Bool, G, x_ints) || throw(BoundsError(G, x_ints))
    radius = G.domain.radius ./ G.dims
    left = G.left
    center = @muladd @. left + radius + (2 * radius) * (x_ints - 1)
    return Box{N,T}(center, radius)
end

key_to_box(G::BoxGrid{N,T,I}, x_ints::CartesianIndex) where {N,T,I} = key_to_box(G, x_ints.I)
key_to_box(G::BoxGrid{N,T,I}, x_ints::Nothing) where {N,T,I} = nothing

"""
    point_to_key(G::BoxGrid, point)

Find the index for the box within a `BoxGrid` 
contatining a point, or `nothing` if the point does 
not lie in the domain. 
"""
@propagate_inbounds function point_to_key(G::BoxGrid{N,T,I}, point) where {N,T,I}
    point in G.domain || return nothing
    xi = (point .- G.left) .* G.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, G, x_ints)
        @debug "something went wrong in point_to_key" point xi x_ints G G.domain
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(G))
    end
    return x_ints
end

"""
    bounded_point_to_key(G::BoxGrid, point)

Find the cartesian index of the nearest box within a 
`BoxGrid` to a point. Conicides with `point_to_key` 
if the point lies in the grid. Default behavior 
is to set `NaN = Inf` if `NaN`s are present in `point`. 
"""
function bounded_point_to_key(G::BoxLayout{B}, point) where {N,T,B<:Box{N,T}}
    center, radius = G.domain
    small_bound = 1 ./ ( 2 .* G.scale )
    left = center .- radius .+ small_bound
    right = center .+ radius .- small_bound
    p = ifelse.(isnan.(point), convert(T, Inf), point)
    p = min.(max.(p, left), right)
    return point_to_key(G, p)
end

"""
    point_to_box(P::BoxLayout, point)

Find the box within a `BoxGrid` containing a point. 
"""
function point_to_box(G::BoxLayout, point)
    x_ints = point_to_key(G, point)
    isnothing(x_ints) && return x_ints
    return @inbounds key_to_box(G, x_ints)
end
