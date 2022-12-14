"""
    SampledBoxMap(map, domain::Box, domain_points, image_points)

Type representing a discretization of a map using sample points. 
Constructors:

* `PointDiscretizedBoxMap`
* `GridBoxMap`
* `MonteCarloBoxMap`
* `AdaptiveBoxMap`

Fields:
* `map`:              map that defines the dynamical system.
* `domain`:           domain of the map, `B`.
* `domain_points`:    the spread of test points to be mapped forward in intersection algorithms.
                      Must have the signature `domain_points(center, radius)` and return 
                      an iterator of points within `Box(center, radius)`. 
* `image_points`:     the spread of test points for comparison in intersection algorithms.
                      Must have the signature `domain_points(center, radius)` and return 
                      an iterator of points within `Box(center, radius)`. 

.
"""
struct SampledBoxMap{N,T,F,D,I} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::D
    image_points::I
end

# wee need a small helper function because of 
# how julia dispatches on `union!`
⊔(set1::AbstractSet, set2::AbstractSet) = union!(set1, set2)
⊔(set1::AbstractSet, object) = union!(set1, (object,))

⊔(d::AbstractDict...) = mergewith!(+, d...)
⊔(d::AbstractDict, p::Pair...) = foreach(q -> d ⊔ q, p)
function ⊔(d::AbstractDict, p::Pair)
    k, v = p
    d[k] = haskey(d, k) ? d[k] + v : v
    d
end

function map_boxes(g::SampledBoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
    P = source.partition
    @floop for box in source
        c, r = box
        for p in g.domain_points(c, r)
            fp = g.map(p)
            hitbox = point_to_box(P, fp)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(fp, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                @reduce(image = S() ⊔ hit)
            end
        end
    end
    return BoxSet(P, image)
end 

function construct_transfers(
        g::SampledBoxMap, boxset::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q<:BoxPartition,S<:OrderedSet}

    P, D = boxset.partition, Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in boxset.set
        box = key_to_box(P, key)
        c, r = box
        for p in g.domain_points(c, r)
            c = g.map(p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                hit in boxset.set || @reduce( variant_keys = S() ⊔ hit )
                @reduce( mat = D() ⊔ ((hit,key) => 1) )
            end
        end
    end
    return mat, variant_keys
end

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain
    n = length(g.domain_points(center, radius))
    print(io, "SampledBoxMap with $(n) sample points")
end

"""
    PointDiscretizedBoxMap(map, domain, points) -> SampledBoxMap

Construct a `SampledBoxMap` that uses the iterator `points` as test points. 
`points` must be an array or iterator of test points within the unit cube 
`[-1,1]^N`. 
"""
function PointDiscretizedBoxMap(map, domain::Box, points)
    domain_points = rescale(points)
    image_points = center
    return SampledBoxMap(map, domain, domain_points, image_points)
end

PointDiscretizedBoxMap(map, P::BoxPartition, points) = PointDiscretizedBoxMap(map, P.domain, points)

"""
    GridBoxMap(map, domain::Box{N,T}; no_of_points::NTuple{N} = ntuple(_->16, N)) -> SampledBoxMap
    GridBoxMap(map, P::BoxPartition{N,T}; no_of_points::NTuple{N} = ntuple(_->16, N)) -> SampledBoxMap

Construct a `SampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `no_of_points`, which is 
a tuple of length equal to the dimension of the domain. 
"""
function GridBoxMap(map, domain::Box{N,T}; no_of_points::NTuple{N}=ntuple(_->no_default(T),N)) where {N,T}
    Δp = 2 ./ no_of_points
    points = SVector{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(no_of_points) ]
    return PointDiscretizedBoxMap(map, domain, points)
end

function GridBoxMap(map, P::BoxPartition{N,T}; no_of_points=ntuple(_->no_default(T),N)) where {N,T}
    GridBoxMap(map, P.domain; no_of_points=no_of_points)
end

"""
    MonteCarloBoxMap(map, domain::Box{N,T}; no_of_points=16*N) -> SampledBoxMap
    MonteCarloBoxMap(map, P::BoxPartition{N,T}; no_of_points=16*N) -> SampledBoxMap

Construct a `SampledBoxMap` that uses `no_of_points` 
Monte-Carlo test points. 
"""
function MonteCarloBoxMap(map, domain::Box{N,T}; no_of_points=no_default(N,T)) where {N,T}
    points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:no_of_points ] 
    return PointDiscretizedBoxMap(map, domain, points) 
end 

function MonteCarloBoxMap(map, P::BoxPartition{N,T}; no_of_points=no_default(N,T)) where {N,T}
    MonteCarloBoxMap(map, P.domain; no_of_points=no_of_points)
end

"""
    AdaptiveBoxMap(f, domain::Box, accel=nothing) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `sample_adaptive` to 
generate test points. 
"""
function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    domain_points = sample_adaptive(f)
    image_points = vertices
    return SampledBoxMap(f, domain, domain_points, image_points)
end

AdaptiveBoxMap(f, P::BoxPartition) = AdaptiveBoxMap(f, P.domain)

Core.@doc raw"""
    approx_lipschitz(f, center::SVector, radius::SVector, accel=nothing) -> Matrix

Compute an approximation of the Lipschitz matrix, 
i.e. the matrix that satisifies 

```math
| f(x) - f(y) | \leq L | x - y | \quad \forall \, x,y \in \text{Box(center, radius)}
```

componentwise. 
"""
function approx_lipschitz(f, center::SVNT{N,T}, radius::SVNT{N,T}) where {N,T}
    L, y = Matrix{T}(undef, N, N), MVector{N,T}(ntuple(_->zero(T),Val(N)))
    fc = f(center)
    for dim in 1:N
        y[dim] = radius[dim]
        fr = f(center .+ y)
        L[:, dim] .= abs.(fr .- fc) ./ radius[dim]
        y[dim] = zero(T)
    end
    if !all(isfinite, L)
        throw(DomainError(
            Inf,
            """The dynamical system diverges within the box. 
            Cannot calculate Lipschitz constant.
            $(Box{N,T}(center, radius))
            """
        ))
    end
    return L
end

"""
    sample_adaptive(f, center::SVector, radius::SVector, accel=nothing)

Create a grid of test points using the adaptive technique 
described in 

Oliver Junge. “Rigorous discretization of subdivision techniques”. In: 
_International Conference on Differential Equations_. Ed. by B. Fiedler, K.
Gröger, and J. Sprekels. 1999. 
"""
function sample_adaptive(f, center::SVNT{N,T}, radius::SVNT{N,T}, alg=LinearAlgebra.QRIteration()) where {N,T}
    L = approx_lipschitz(f, center, radius)
    _, σ, Vt = svd(L, alg=alg)
    n = ntuple(i->ceil(Int, σ[i]), Val(N))
    h = 2.0 ./ (n .- 1)
    points = Iterators.map(CartesianIndices(n)) do i
        p  = T[ n[k] == 1 ? 0 : (i[k] - 1) * h[k] - 1 for k in 1:N ]
        p .= Vt'p
        @muladd p .= center .+ radius .* p
        sp = SVector{N,T}(p)
    end
    return points
end

sample_adaptive(f) = (center, radius) -> sample_adaptive(f, center, radius)

no_default(T) = Int(pick_vector_width(T))
no_default(N, T) = 4 * N * no_default(T)
