abstract type BoxMap end
"""
Transforms a `map` defined on ℝᴺ to a `BoxMap` defined on BoxSets

`map`:              map that defines the dynamical system

`domain_points`:    the spread of test points to be mapped forward in intersection algorithms.
                    (scaled to fit a box with unit radii)

`image_points`:     the spread of test points for comparison in intersection algorithms.
                    (scaled to fit a box with unit radii)

""" # TODO: see if domain is redundant in struct
struct SampledBoxMap{F,N,T,P,I,B} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::P
    image_points::I
    accepts_unroll::Val{B}
end

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

function PointDiscretizedMap(map, domain, points::AbstractArray) 
    function domain_points(center, radius); points end
    function domain_points(::Box); points end
    function image_points(center, radius); center end
    function image_points(b::Box); b.center end
    return SampledBoxMap(map, domain, domain_points, image_points, Val(false))
end

function BoxMap(map, domain::Box{N,T}; no_of_points::Int=16*N) where {N,T}
    points = [ SVector{N, T}(2.0*rand(N).-1.0 ...) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points) 
end

function BoxMap(map, P::BoxPartition{N,T}; adaptive=false, no_of_points::Int=16*N) where {N,T}
    adaptive  ?  AdaptiveBoxMap(map, P.domain) : BoxMap(map, P.domain, no_of_points=no_of_points)
end

function sample_adaptive(Df, center::SVector{N,T}) where {N,T}  # how does this work?
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ) 
    h = 2.0./(n.-1)
    points = Array{SVector{N,T}}(undef, ntuple(i->n[i], N))
    # length(points) == prod(n[i] for i=1:N)
    for i in CartesianIndices(points)
        points[i] = ntuple(k -> n[k]==1 ? 0.0 : (i[k]-1)*h[k]-1.0, N)
        points[i] = Vt'*points[i]
    end   
    @debug points
    return points 
end

function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    function domain_points(center, radius); sample_adaptive(Df, center) end
    function domain_points(b::Box); sample_adaptive(Df, b.center) end

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end
    # calculates the vertices of each box
    function image_points(center, radius); vertices end
    function image_points(::Box); vertices end
    return SampledBoxMap(f, domain, domain_points, image_points, Val(false))
end

function scaled_domain_points(g::SampledBoxMap{F,N,T,P,I,B}, b::Box{N,T}) where {F,N,T,P,I,B}
    points = reinterpret(T, g.domain_points(b.center, b.radius))
    n = floor(Int, length(points) / N)
    @inbounds @muladd for i in 0:n-1, j in 1:N
        points[N*i + j] = b.center[j] + b.radius[j] * points[N*i + j]
    end
    return reinterpret(SVector{N, T}, points)
end


function map_boxes(g::BoxMap, source::BoxSet)
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = g.map(c.+r.*p)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    BoxSet(P, union(image...))
end 

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)
