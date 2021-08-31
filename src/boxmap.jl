abstract type BoxMap end

struct SampledBoxMap{F,N,T,P,I} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::P
    image_points::I    
end

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

function PointDiscretizedMap(map, domain, points::AbstractArray) 
    domain_points = (center, radius) -> points
    image_points = (center, radius) -> center
    return SampledBoxMap(map, domain, domain_points, image_points)
end

function BoxMap(map, domain::Box{N,T}; no_of_points=20*N) where {N,T}
    points = [ tuple(2.0*rand(N).-1.0 ...) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points) 
end

function sample_adaptive(Df, center::SVector{N,T}) where {N,T}
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ) 
    h = 2.0./(n.-1)
    points = Array{SVector{N,T}}(undef, ntuple(i->n[i], N))
    for i in CartesianIndices(points)
        points[i] = ntuple(k -> n[k]==1 ? 0.0 : (i[k]-1)*h[k]-1.0, N)
        points[i] = Vt'*points[i]
    end   
    @show points
    return points 
end

function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    domain_points = (center, radius) -> sample_adaptive(Df, center)

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end   
    image_points = (center, radius) -> vertices
    return SampledBoxMap(f, domain, domain_points, image_points)
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



