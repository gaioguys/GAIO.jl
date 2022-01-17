using LinearAlgebra, SparseArrays, StaticArrays, Base.Threads, 
      LoopVectorization, MuladdMacro, BenchmarkTools


function map_boxes_1(g::BoxMap, source::BoxSet)
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
                @inbounds push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(P, union(image...))
end 

function map_boxes_2(g::BoxMap, source::BoxSet)
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]
    domain_points = g.domain_points(g.domain)
    points = [ similar(domain_points) for _ in 1:nthreads() ]
    gmap(r, p, c) = g.map(muladd.(r, p, c))
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        map!(p -> gmap(r, p, c), points[threadid()], domain_points)
        union!(image[threadid()], map(p -> point_to_key(P, p), points[threadid()]))
    end
    return BoxSet(P, delete!(union(image...), nothing))
end 

"""
`warning!` requires that the given map f accepts an `unroll` kwarg, ie

```julia
function f(points; unroll=1)
    @assert (n = length(points)) % unroll == 0
    fp = similar(p)
    for i in 1:n
        fp[i] = "some map on points[i]..."
    end
    return fp
```

Ideally, the for loop in `f` would use `@fastmath`, `@muladd`, `@turbo` 
(from the packages Base, MuladdMacro, LoopVectorization) in order to take advantage 
of loop structure.

`TODO: autogenerate new unrolled map from given map? 
That might defeat the point - to create genericity, one is left with only
one option: the unrolled map is just a for loop over the points, same as before`
"""
function map_boxes_unroll(g::SampledBoxMap{F,N,T,P,I}, source::BoxSet; unroll=1) where {F,N,T,P,I}
    part, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]
    L = N * unroll

    @inbounds @threads for key in keys
        box = key_to_box(part, key)
        c, r = box.center, box.radius
        points = reduce(vcat, g.domain_points(c, r))
        n = length(points)
        k, rem = divrem(n, L)
        
        @fastmath @turbo for i in 0:N:n-N
            for d in 1:N
                ind = i+d
                @muladd points[ind] = c[ind] + r[ind] * points[ind]
            end
        end

        if k != 0
            for i in 0:L:(k-1)*L
                fp = g.map(points[i+1:i+L]; unroll=unroll)

                for j in 0:N:L-N
                    hit = point_to_key(part, fp[j+1:j+N])

                    if !isnothing(hit)
                        push!(image[threadid()], hit)
                    end
                end
            end
        else
            @debug """Unroll factor too large for number of points. 
            Defaulting to single-point loops.""" nr_of_points=n/N unroll
        end 

        if rem != 0
            @debug """Unroll complete, but points remain. 
            Remainder of points are calulated in single-point-loops.""" nr_of_points=n/N unroll rem
            for i in k*L:N:n-N
                fp = g.map(points[i+1:i+N], unroll=1)
                hit = point_to_key(part, fp)

                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end 


# more difficult example: Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
F = BoxMap(f, P)

const x_eq = SVector{3, Float64}([sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1])
function S_init()
    points = [SVector{3, Float64}(x_eq .+ SVector{3, Float64}(15 .* rand(3))) for _ in 1:100]
    BoxSet(P, Set(point_to_key(P, point) for point in points))
end

map_boxes_1(F, S_init()); map_boxes_2(F, S_init());
@benchmark map_boxes_1($F, S) setup=(S=S_init())
@benchmark map_boxes_2($F, S) setup=(S=S_init())







function map_boxes_unroll(g::SampledBoxMap{F,N,T,P,I}, source::BoxSet; unroll=1) where {F,N,T,P,I}
    part, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]

    #= @inbounds =# @threads for key in keys
        box = key_to_box(part, key)
        points = scaled_domain_points(g, box)
        n = length(points)
        k, rem = divrem(n, unroll)

        if k != 0
            for i in 0:unroll:(k-1)*unroll
                fp = g.map(points[i+1:i+unroll])

                for p in fp
                    hit = point_to_key(part, p)

                    if !isnothing(hit)
                        push!(image[threadid()], hit)
                    end
                end
            end
        else
            @debug """Unroll factor too large for number of points. 
            Defaulting to single-point loops.""" n unroll
        end 

        if rem != 0
            @debug """Unroll complete, but points remain. 
            Remainder of points are calulated in single-point-loops.""" n unroll rem
            for i in k*unroll+1:n
                fp = g.map(points[i])
                hit = point_to_key(part, fp)

                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end 
