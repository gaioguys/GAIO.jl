abstract type BoxMap end

"""
    SampledBoxMap(map, domain::Box, domain_points, image_points, acceleration)

Transforms a ``map: Q → Q`` defined on points in the box ``Q ⊂ ℝᴺ`` to a `SampledBoxMap` defined 
on `Box`es. 

Constructors:
* `BoxMap`
* `PointDiscretizedMap`
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

@inbounds @muladd function map_boxes(g::SampledBoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
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

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

"""
    PointDiscretizedMap(map, domain, points, accel=nothing) -> SampledBoxMap

Construct a `SampledBoxMap` that uses the iterator `points` as test points. 
`points` must be an array or iterator of test points within the unit cube 
``[-1,1]^N``. 
"""
function PointDiscretizedMap(map, domain, points)
    domain_points = rescale(points)
    image_points = center
    return SampledBoxMap(map, domain, domain_points, image_points)
end

function GridMap(map, domain::Box{N,T}; no_of_points::NTuple{N}=ntuple(_->4*pick_vector_width(T),N)) where {N,T}
    Δp = 2 ./ no_of_points
    points = NTuple{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(no_of_points) ]
    return PointDiscretizedMap(map, domain, points)
end

function GridMap(map, P::BoxPartition{N,T}; no_of_points=ntuple(_->4*pick_vector_width(T),N)) where {N,T}
    GridMap(map, P.domain; no_of_points=no_of_points)
end

function GridMap(map, domain::Box{N,T}; no_of_points::Integer) where {N,T}
    n = ceil(Int, no_of_points ^ 1/n)
    GridMap(map, domain; no_of_points=ntuple(_->n, N))
end

"""
    MonteCarloMap(map, domain::Box{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) -> SampledBoxMap
    MonteCarloMap(map, P::BoxPartition{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `no_of_points` Monte-Carlo 
test points. 
"""
function MonteCarloMap(map, domain::Box{N,T}; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    points = NTuple{N,T}[ 2*rand(T,N).-1 for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points) 
end 

function MonteCarloMap(map, P::BoxPartition{N,T}; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    BoxMap(map, P.domain, accel; no_of_points=no_of_points)
end

"""
    AdaptiveBoxMap(f, domain::Box, accel=nothing) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `sample_adaptive` to generate 
test points. 
"""
function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    domain_points = sample_adaptive(f, accel)
    image_points = vertices
    return SampledBoxMap(f, domain, domain_points, image_points, accel)
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
        @error(
            """The dynamical system diverges within the box. 
            Cannot calculate Lipschitz constant.""", 
            box=Box{N,T}(center, radius)
        )
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
function sample_adaptive(f, center::SVNT{N,T}, radius::SVNT{N,T}) where {N,T}
    L = approx_lipschitz(f, center, radius, accel)
    _, σ, Vt = svd(L)
    n = ceil.(Int, σ)
    h = 2.0 ./ (n .- 1)
    points = Iterators.map(CartesianIndices(ntuple(k -> n[k], Val(N)))) do i
        p  = [n[k] == 1 ? zero(T) : (i[k] - 1) * h[k] - 1 for k in 1:N]
        p .= Vt'p
        @muladd p .= center .+ radius .* p
        sp = SVector{N,T}(p)
    end
    return points
end

sample_adaptive(f, accel=nothing) = (center, radius) -> sample_adaptive(f, center, radius)
