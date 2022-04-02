abstract type BoxMap end
"""
Transforms a `map: B → C, B ⊂ ℝᴺ` to a `SampledBoxMap` defined on `BoxSet`s

`map`:              map that defines the dynamical system.

`domain`:           domain of the map, `B`.

`domain_points`:    the spread of test points to be mapped forward in intersection algorithms.
                    (scaled to fit a box with unit radii)

`image_points`:     the spread of test points for comparison in intersection algorithms.
                    (scaled to fit a box with unit radii)

`acceleration`:     `WARNING UNFINISHED` Whether to use optimized functions in intersection algorithms.
                    Accepted values: `nothing`, `Val(:cpu)`, `Val(:gpu)`.

"""
struct SampledBoxMap{N,T,B} <: BoxMap
    map
    domain::Box{N,T}
    domain_points
    image_points
    acceleration::B
end

function SampledBoxMap(map, domain, domain_points, image_points)
    SampledBoxMap(map, domain, domain_points, image_points, nothing)
end
function SampledBoxMap(map, domain::Box{N,T}, domain_points, image_points, accel::Symbol) where {N,T}
    SampledBoxMap(map, domain, domain_points, image_points, Val(accel))
end

function Base.show(io::IO, g::BoxMap)
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

function PointDiscretizedMap(map, domain, points::AbstractArray, accel=nothing)
    domain_points(center, radius) =  points
    image_points(center, radius) = center
    return SampledBoxMap(map, domain, domain_points, image_points, accel)
end

function BoxMap(map, domain::Box{N,T}, accel=nothing; no_of_points::Int=16*N) where {N,T}
    points = [ SVector{N,T}(2.0*rand(T,N).-1.0) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points, accel) 
end

function BoxMap(map, P::BoxPartition{N,T}, accel=nothing; no_of_points::Int=16*N) where {N,T}
    BoxMap(map, P.domain, accel; no_of_points=no_of_points)
end

function sample_adaptive(Df, center::SVector{N,T}) where {N,T}  # how does this work?
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ) 
    h = 2.0./(n.-1)
    points = Array{SVector{N,T}}(undef, ntuple(i->n[i], N))
    for i in CartesianIndices(points)
        points[i] = ntuple(k -> n[k]==1 ? 0.0 : (i[k]-1)*h[k]-1.0, N)
        points[i] = Vt'*points[i]
    end   
    @debug points
    return points 
end

function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    domain_points(center, radius) = sample_adaptive(Df, center)

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end
    # calculates the vertices of each box
    image_points(center, radius) = vertices
    return SampledBoxMap(f, domain, domain_points, image_points)
end

function map_boxes(g::BoxMap, source::BoxSet)
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = @muladd p .* r .+ c
            fp = g.map(fp)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    BoxSet(P, union(image...))
end 

function map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = deepcopy(g.domain_points(c, r))
        points_vec = reinterpret(T, points)
        @turbo for i in 0:N:length(points_vec)-N, j in 1:N
            points_vec[i+j] = @muladd points_vec[i+j] * r[j] + c[j]
        end
        points_vec .= g.map(points_vec)
        points = reinterpret(SVector{N,T}, points_vec)
        for p in points
            hit = point_to_key(P, p)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    BoxSet(P, union(image...))
end 

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)
