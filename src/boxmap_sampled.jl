"""
    BoxMap(:sampled, map, domain::Box, domain_points, image_points)

Type representing a discretization of a map using sample points. 

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
struct SampledBoxMap{N,T} <: BoxMap
    map
    domain::Box{N,T}
    domain_points
    image_points
end

# Constructors

"""
    BoxMap(:pointdiscretized, map, domain, points) -> SampledBoxMap

Construct a `SampledBoxMap` that uses the iterator `points` as test points. 
`points` must be an array or iterator of test points within the unit cube 
`[-1,1]^N`. 
"""
function PointDiscretizedBoxMap(map, domain::Box, points)
    domain_points = rescale(points)
    image_points = center
    return SampledBoxMap(map, domain, domain_points, image_points)
end

PointDiscretizedBoxMap(map, P::BoxLayout, points) = PointDiscretizedBoxMap(map, P.domain, points)

"""
    BoxMap(:grid, map, domain::Box{N}; n_points::NTuple{N} = ntuple(_->16, N)) -> SampledBoxMap

Construct a `SampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `n_points`, which is 
a tuple of length equal to the dimension of the domain. 
"""
function GridBoxMap(map, domain::Box{N,T}; n_points::NTuple{N}=ntuple(_->4,N)) where {N,T}
    Δp = 2 ./ n_points
    points = SVector{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(n_points) ]
    return PointDiscretizedBoxMap(map, domain, points)
end

function GridBoxMap(map, P::Q; n_points=ntuple(_->4,N)) where {N,T,Q<:BoxLayout{Box{N,T}}}
    GridBoxMap(map, P.domain; n_points=n_points)
end

"""
    BoxMap(:montecarlo, map, domain::Box{N}; n_points=16*N) -> SampledBoxMap

Construct a `SampledBoxMap` that uses `n_points` 
Monte-Carlo test points. 
"""
function MonteCarloBoxMap(map, domain::Box{N,T}; n_points=16*N) where {N,T}
    points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:n_points ] 
    return PointDiscretizedBoxMap(map, domain, points) 
end 

function MonteCarloBoxMap(map, P::Q; n_points=16*N) where {N,T,Q<:BoxLayout{Box{N,T}}}
    MonteCarloBoxMap(map, P.domain; n_points=n_points)
end

"""
    BoxMap(:adaptive, f, domain::Box) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `sample_adaptive` to 
generate test points as described in 

Oliver Junge. “Rigorous discretization of subdivision techniques”. In: 
_International Conference on Differential Equations_. Ed. by B. Fiedler, K.
Gröger, and J. Sprekels. 1999. 
"""
function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    domain_points = sample_adaptive(f)
    image_points = vertices
    return SampledBoxMap(f, domain, domain_points, image_points)
end

AdaptiveBoxMap(f, P::BoxLayout) = AdaptiveBoxMap(f, P.domain)

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
        throw(NumericalError(center, radius, DIVERGED_MSG))
    end
    return L
end

"""
    sample_adaptive(f, center::SVector, radius::SVector)

Create a grid of test points using the adaptive technique 
described in 

Oliver Junge. “Rigorous discretization of subdivision techniques”. In: 
_International Conference on Differential Equations_. Ed. by B. Fiedler, K.
Gröger, and J. Sprekels. 1999. 
"""
function sample_adaptive(f, center::SVNT{N,T}, radius::SVNT{N,T}; alg=LinearAlgebra.QRIteration()) where {N,T}
    L = approx_lipschitz(f, center, radius)
    _, σ, Vt = svd(L, alg=alg)

    if !all(typemin(Int) .< σ .< typemax(Int))
        throw(NumericalError(center, radius, SVD_MSG))
    end

    n = ntuple(i -> ceil(Int, σ[i]), Val(N))
    h = convert(T, 2) ./ (n .- 1)
    points = Iterators.map(CartesianIndices(n)) do i
        p  = T[ n[k] == 1 ? zero(T) : (i[k] - 1) * h[k] - 1 for k in 1:N ]
        p .= Vt'p
        @muladd p .= center .+ radius .* p
        sp = SVector{N,T}(p)
    end

    return points
end

function sample_adaptive(f)
    domain_points(center, radius) = sample_adaptive(f, center, radius)
end

struct NumericalError{B} <: Exception
    box::B
    msg::String
end

function NumericalError(center, radius::SVNT{N,T}, msg) where {N,T}
    NumericalError(Box{N,T}(center, radius), msg)
end

function Base.showerror(io::IO, err::NumericalError)
    println(io, "NumericalError within the box $(err.box): ")
    println(io, err.msg)
end

const DIVERGED_MSG = """
The dynamical system diverges. 
Cannot calculate Lipschitz constant.  
Make sure that the dynamical system is well-defined 
or use a different `BoxMap` discretization. 
"""

const SVD_MSG = """
The dynamical system is too expansive. 
Cannot calculate a test point grid. 
Make sure that the dynamical system is well-defined 
or use a different `BoxMap` discretization. 
"""

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain
    n = length(g.domain_points(center, radius))
    print(io, "SampledBoxMap with $(n) sample points")
end

function typesafe_map(g::SampledBoxMap{N,T}, x::SVNT{N}) where {N,T}
    convert(SVector{N,T}, g.map(x))
end

# BoxMap API

function map_boxes(
        g::SampledBoxMap, source::BoxSet{B,Q,S}
    ) where {B,Q,S}

    P = source.partition
    @floop for box in source
        c, r = box
        for p in g.domain_points(c, r)
            c = typesafe_map(g, p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                @reduce() do (image = S(); hit)     # Initialize empty key set
                    image = image ⊔ hit             # Add hit key to image
                end
            end
        end
    end
    return BoxSet(P, image::S)
end 

function construct_transfers(
        g::SampledBoxMap, domain::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q,S}

    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in keys(domain)
        box = key_to_box(P, key)
        c, r = box
        for p in g.domain_points(c, r)
            c = typesafe_map(g, p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                weight = (hit,key) => 1
                @reduce() do (image = S(); hit), (mat = D(); weight)    # Initialize empty key set and dict-of-keys sparse matrix
                    image = image ⊔ hit                                 # Add hit key to image
                    mat = mat ⊔ weight                                  # Add weight to mat[hit, key]
                end
            end
        end
    end
    return mat::D, BoxSet(P, image::S)
end

function construct_transfers(
        g::SampledBoxMap, domain::BoxSet{R,Q}, codomain::BoxSet{U,H}
    ) where {N,T,R<:Box{N,T},Q,U,H}

    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}
    @floop for key in keys(domain)
        box = key_to_box(P1, key)
        c, r = box
        for p in g.domain_points(c, r)
            c = typesafe_map(g, p)
            hitbox = point_to_box(P2, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P2, ip)
                hit in codomain.set || continue
                weight = (hit,key) => 1
                @reduce() do (mat = D(); weight)     # Initialize dict-of-keys sparse matrix
                    mat = mat ⊔ weight               # Add weight to mat[hit, key]
                end
            end
        end
    end
    return mat::D
end
