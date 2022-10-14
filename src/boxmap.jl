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
* `acceleration`:     Whether to use optimized functions in intersection algorithms.
                      Accepted values: `nothing`, `BoxMapCPUCache`, `BoxMapGPUCache`.
                      `BoxMapGPUCache` does nothing unless you have a CUDA capable gpu.

.
"""
struct SampledBoxMap{A,N,T,F,D,I} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::D
    image_points::I
    acceleration::A
end

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

@inbounds @muladd function map_boxes(g::SampledBoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
    P = source.partition
    @floop for box in source
        c, r = box.center, box.radius
        for p in g.domain_points(c, r)
            fp = g.map(p)
            hitbox = point_to_box(P, fp)
            isnothing(hitbox) && continue
            r = hitbox.radius
            for ip in g.image_points(fp, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                @reduce(image = union!(S(), hit))
            end
        end
    end
    return BoxSet(P, image)
end 

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)

"""
    PointDiscretizedMap(map, domain, points, accel=nothing) -> SampledBoxMap

Construct a `SampledBoxMap` that uses the iterator `points` as test points. 
`points` must be an array or iterator of test points within the unit cube 
``[-1,1]^N``. 
"""
function PointDiscretizedMap(map, domain, points, accel=nothing)
    domain_points = rescale(points)
    image_points = center
    return SampledBoxMap(map, domain, domain_points, image_points, accel)
end

function PointDiscretizedMap(map, domain, points, accel::Symbol)
    return PointDiscretizedMap(map, domain, points, Val(accel))
end

"""
    BoxMap(map, domain::Box{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) -> SampledBoxMap
    BoxMap(map, P::BoxPartition{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `no_of_points` Monte-Carlo 
test points. 
"""
function BoxMap(map, domain::Box{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    points = [ tuple(2f0*rand(T,N).-1f0 ...) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points, accel) 
end 

function BoxMap(map, P::BoxPartition{N,T}, accel=nothing; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    BoxMap(map, P.domain, accel; no_of_points=no_of_points)
end

"""
    AdaptiveBoxMap(f, domain::Box, accel=nothing) -> SampledBoxMap

Construct a `SampledBoxMap` which uses `sample_adaptive` to generate 
test points. 
"""
function AdaptiveBoxMap(f, domain::Box, accel=nothing)
    domain_points = sample_adaptive(f, accel)
    image_points = vertices
    return SampledBoxMap(f, domain, domain_points, image_points, nothing)
end

AdaptiveBoxMap(f, P::BoxPartition, accel=nothing) = AdaptiveBoxMap(f, P.domain, Val(accel))
AdaptiveBoxMap(f, domain::Box, accel::Symbol) = AdaptiveBoxMap(f, domain, Val(accel))
AdaptiveBoxMap(f, P::BoxPartition, accel::Symbol) = AdaptiveBoxMap(f, P.domain, Val(accel))

"""
    sample_adaptive(f, center::SVector, radius::SVector, accel=nothing)

Create a grid of test points using the adaptive technique 
described in 

Oliver Junge. “Rigorous discretization of subdivision techniques”. In: 
_International Conference on Differential Equations_. Ed. by B. Fiedler, K.
Gröger, and J. Sprekels. 1999. 
"""
function sample_adaptive(f, center::SVNT{N,T}, radius::SVNT{N,T}, accel=nothing) where {N,T}
    L, y, n, h = zeros(T,N,N), MVector{N,T}(zeros(T,N)), MVector{N,Int}(undef), MVector{N,T}(undef)
    fc = f(center)
    for dim in 1:N
        y[dim] = radius[dim]
        fr = f(center .+ y)
        L[:, dim] .= abs.(fr .- fc) ./ radius[dim]
        y[dim] = zero(T)
    end
    all(isfinite, L) || @error "The dynamical system diverges within the box. Cannot calculate Lipschitz constant." box=Box{N,T}(center, radius)
    _, σ, Vt = svd(L)
    n .= ceil.(Int, σ)
    h .= 2.0 ./ (n .- 1)
    points = Iterators.map(CartesianIndices(ntuple(k -> n[k], Val(N)))) do i
        p = [n[k] == 1 ? zero(T) : (i[k] - 1) * h[k] - 1 for k in 1:N]
        p .= Vt'p
        @muladd p .= center .+ radius .* p
        sp = SVector{N,T}(p)
    end
    return points
end

sample_adaptive(f, accel=nothing) = (center, radius) -> sample_adaptive(f, center, radius, accel)
