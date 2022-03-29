using GAIO
using Test
using LoopVectorization, MuladdMacro, Base.Threads, StaticArrays

a, b = 1.4, 0.3
function f(x)
    return (1 - a*(x[1])^2 + x[2], b*(x[1]))
end
function f2(x)
    dx = similar(x)
    @turbo for i in 0 : 2 : length(x) - 2
        dx[i+1] = 1 - a*(x[i+1])^2 + x[i+2]
        dx[i+2] = b*(x[1])
    end
    return dx
end


center, radius, size = (0, 0), (3, 3), (256, 256)
P = BoxPartition(Box(center, radius), size)

# simple boxmap
F = BoxMap(f, P, no_of_points=64)
# boxmap with the same test points
F2 = SampledBoxMap(f2, F.domain, F.domain_points, F.image_points, Val(:cpu))

x = (1.5, 1.5)
box = key_to_box(P, point_to_key(P, x))
c, r = box.center, box.radius
@assert F.domain_points(c, r) == F2.domain_points(c, r)


function map_boxes(g::SampledBoxMap{N,T,Nothing}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    out_points = fill( SVector{N,T}[], nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = @muladd p .* r .+ c
            fp = g.map(fp)
            push!(out_points[threadid()], fp)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(P, union(image...)), vcat(out_points...)
end 

function map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    out_points = fill( SVector{N,T}[], nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = deepcopy(g.domain_points(c, r))
        points_vec = reinterpret(T, points)
        @turbo @. points_vec = points_vec * r + c
        points_vec .= g.map(points_vec)
        points = reinterpret(SVector{N,T}, points_vec)
        push!(out_points[threadid()], points...)
        for p in points
            hit = point_to_key(P, p)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(P, union(image...)), vcat(out_points...)
end 

@test issetequal(map_boxes(F, P[x])[2], map_boxes(F, P[x])[2])