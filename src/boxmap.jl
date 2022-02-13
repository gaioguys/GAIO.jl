abstract type BoxMap end

"""
Transforms a `map` defined on ℝᴺ to a `BoxMap` defined on BoxSets

`map`:              map that defines the dynamical system

`domain_points`:    the spread of test points to be mapped forward in intersection algorithms.
                    (scaled to fit a box with unit radii)

`image_points`:     the spread of test points for comparison in intersection algorithms.
                    (scaled to fit a box with unit radii)

`unroll`:           method used to unroll `map` for performance. Can be 
                    `Val(:none)`, `Val(:cpu)`, `Val(:gpu)`

"""
struct SampledBoxMap{N,T,B} <: BoxMap
    map::Function
    domain::Box{N,T}
    domain_points::Function
    image_points::Function
    unroll::Val{B}
    
    function SampledBoxMap(map, domain, domain_points, image_points, unroll)
        new(map, domain, domain_points, image_points, unroll)
    end
    function SampledBoxMap(map, domain::Box{N,T}, domain_points, image_points, ::Val{:cpu}) where {N,T}
        if hasmethod(map, (AbstractVector,), (:simd,))
            return new(map, domain, domain_points, image_points, Val{:cpu}())
        else
            unrolled_map = @generate_unrolled map N
            simd_map = @generate_simd unrolled_map N
            return new(simd_map, domain, domain_points, image_points, Val{:cpu}())
        end
    end
end


function Base.show(io::IO, g::SampledBoxMap{N,T,B}) where {N,T,B}
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius))
    print(io, "BoxMap in $N dimensions with $n sample points.\n eltype: $T\n unroll type: $B")
end

Base.eltype(::SampledBoxMap{N,T,B}) where {N,T,B} = T

function PointDiscretizedMap(map, domain, points::AbstractArray, unroll) 
    domain_points = (center, radius) -> points
    image_points = (center, radius) -> center
    return SampledBoxMap(map, domain, domain_points, image_points, unroll)
end

function BoxMap(map, domain::Box{N,T}; no_of_points::Int=16*N, unroll=Val{:none}()) where {N,T}
    points = [ SVector{N,T}(2.0*rand(N).-1.0) for _ = 1:no_of_points ] 
    return PointDiscretizedMap(map, domain, points, unroll) 
end

function BoxMap(map, P::BoxPartition{N,T}; no_of_points::Int=16*N, unroll=Val{:none}()) where {N,T}
    BoxMap(map, P.domain, no_of_points=no_of_points, unroll=unroll)
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

function AdaptiveBoxMap(f, domain::Box{N,T}; unroll=Val{:none}()) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    domain_points = (center, radius) -> sample_adaptive(Df, center)

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end
    # calculates the vertices of each box
    image_points = (center, radius) -> vertices
    return SampledBoxMap(f, domain, domain_points, image_points, unroll)
end

# TODO: only allow inplace functions?
macro generate_unrolled(f, N)
    quote
        if hasmethod($(esc(f)), (AbstractVector, AbstractVector))
            function unrolled_function(u)
                @assert (n = length(u)) % $N == 0
                du = similar(u)
                @inbounds for i in 0 : $N : n - $N
                    $(esc(f))(
                        @view( du[i+1:i+$N] ), 
                                u[i+1:i+$N]
                    )
                end
                return du
            end
        else
            function unrolled_function(u)
                @assert (n = length(u)) % $N == 0
                du = similar(u)
                @inbounds for i in 0 : $N : n - $N
                    du[i+1:i+$N] .= $(esc(f))(u[i+1:i+$N])
                end
                return du
            end
        end
        unrolled_function
    end
end

macro generate_simd(f, N)
    quote
        @inbounds @muladd function simd_function(u::Vector{SVector{$N,T}}; simd=Int(pick_vector_width(T))) where T
            n = length(u)
            m, r = divrem(n, simd)
            @assert r == 0
            u_new = reshape(permutedims(reshape(reinterpret(T, u), ($N, n))), ($N * n,))
            idx = VecRange{simd}(1)
            u_tmp = Vector{Vec{simd,T}}(
                reshape(
                    [u_new[idx + simd * i + n * j]
                     for j in 0 : $N - 1, i in  0 : m - 1],
                    ($N * m,)
                )
            )
            u_tmp .= $(esc(f))(u_tmp)
            for i in  0 : m - 1, j in 0 : $N - 1
                u_new[idx + simd * i + n * j] = u_tmp[$N * i + j + 1]
            end
            u .= reinterpret(SVector{$N,T}, reshape(permutedims(reshape(u_new, (n, $N))), ($N * n,)))
            return u
        end
        simd_function
    end
end

function map_boxes(g::SampledBoxMap{N,T,:none}, source) where {N,T}
    part, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    @threads for key in keys
        box = key_to_box(part, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        map!(p -> muladd.(r, p, c), points, points)
        points .= (g.map).(points)
        for p in points
            hit = point_to_key(part, p)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(part, union(image...))
end

function map_boxes(g::SampledBoxMap{N,T,:cpu}, source) where {N,T}
    part, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    simd_length = Int(HostCPUFeatures.pick_vector_width(T))
    @threads for key in keys
        box = key_to_box(part, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        map!(p -> muladd.(r, p, c), points, points)
        s = simd_length
        i, k, rem = 0, 0, length(points)
        idx = 1:2
        while rem > 0
            k, rem = divrem(rem, s)
            idx = i + 1  :  i + k * s
            points[idx] .= g.map(points[idx], simd=s)
            s = s ÷ 2
            i = idx[end]
        end
        for p in points
            hit = point_to_key(part, p)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(part, union(image...))
end

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)
