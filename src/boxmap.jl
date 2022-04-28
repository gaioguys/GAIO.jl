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
                    `Val(:gpu)` currently does nothing.

"""
struct SampledBoxMap{F,N,T,D,I,B} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::D
    image_points::I
    acceleration::B
end

function Base.show(io::IO, g::SampledBoxMap)
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap with $(n) sample points")
end

function PointDiscretizedMap(map, domain, points, accel=nothing)
    domain_points = (center, radius) -> points
    image_points = (center, radius) -> center
    return SampledBoxMap(map, domain, domain_points, image_points, accel)
end

function PointDiscretizedMap(map, domain, points::V, accel::Val{:cpu}) where {T,V<:AbstractArray{T}}
    n, simd = length(points), Int(pick_vector_width(T))
    if n % simd != 0
        throw(DimensionMismatch("Number of test points $n is not divisible by $T SIMD capability $simd"))
    end
    gathered_points = copy(tuple_vgather_lazy(points, simd))
    domain_points = (center, radius) -> gathered_points
    image_points = (center, radius) -> center
    return SampledBoxMap(map, domain, domain_points, image_points, accel)
end

function PointDiscretizedMap(map, domain, points, accel::Symbol)
    return PointDiscretizedMap(map, domain, points, Val(accel))
end

function BoxMap(map, domain::Box{N,T}, accel=nothing; no_of_points::Int=4*N*pick_vector_width(T)) where {N,T}
    points = [ tuple(2.0*rand(T,N).-1.0 ...) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points, accel) 
end 

function BoxMap(map, P::BoxPartition{N,T}, accel=nothing; no_of_points::Int=4*N*pick_vector_width(T)) where {N,T}
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
    domain_points = (center, radius) -> sample_adaptive(Df, center)

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end
    # calculates the vertices of each box
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
            fp = g.map(p .* r .+ c)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    BoxSet(P, union(image...))
end 

# Julia doesn't compile the SIMD accelerated version as efficiently
# as the normal version, so we must manually preallocate memory
function map_boxes(g::SampledBoxMap{F,N,T,D,I,Val{:cpu}}, source::BoxSet) where {F,N,T,D,I}
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for _ in  1:nthreads() ]
    domain_points = g.domain_points(P.domain.center, P.domain.radius)
    simd = get_vector_width(domain_points)
    image_points = Vector{T}(undef, N*simd*nthreads())
    ip  = tuple_vgather_lazy(image_points, simd)
    ipo = reinterpret(SVector{N,T}, image_points)
    @threads for key in keys
        tid  = threadid()
        stid = (tid - 1) * simd + 1 : (tid - 1) * simd + N
        box  = key_to_box(P, key)
        c, r = box.center, box.radius
        for p in domain_points
            ip[tid] = @muladd p .* r .+ c
            ip[tid] = g.map(ip[tid])
            for i in stid
                hit = point_to_key(P, ipo[i])
                if !isnothing(hit)
                    push!(image[tid], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)
