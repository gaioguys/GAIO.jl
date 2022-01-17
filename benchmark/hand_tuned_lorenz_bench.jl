using LinearAlgebra
using SparseArrays
using StaticArrays
using GeometryBasics
using Arpack
using Base.Threads
using LoopVectorization
using MuladdMacro

import Base.round
Base.round(::Type{Int64}, n::Int64, ::RoundingMode) = n

using BenchmarkTools

include("../src/box.jl")
abstract type AbstractBoxPartition{B <: Box} end
include("../src/partition_regular.jl")
include("../src/partition_tree.jl")
include("../src/boxset.jl")
include("../src/boxmap.jl")
include("../src/boxfun.jl")  
#include("../src/transfer_operator.jl")
include("../src/algorithms.jl")
#include("../src/plot.jl")

# TODO: change to SVector style to not need flattening

const σ, ρ, β = 10.0, 28.0, 0.4
function lorenz_unroll!(du, u, unroll)
    @assert (L = 3 * unroll) == length(u)
    @fastmath @muladd @simd for i in 1:3:L-2
        du[i]   =     σ * u[i+1]  -     σ * u[i]
        du[i+1] =     ρ * u[i]    -  u[i] * u[i+2] - u[i+1]
        du[i+2] =  u[i] * u[i+1]  -     β * u[i+2]
    end
end
@benchmark lorenz_unroll!(fill(10., 300), fill(10., 300), 100)

""" 
RK4 Tableu (autonomous)
 0    0    0    0
1/2   0    0    0
 0   1/2   0    0
 0    0    1    0
------------------
1/6  1/3  1/3  1/6
"""
const si = 1/6
@fastmath @muladd function lorenz_rk4_unroll(du, dx, u_tmp, u, τ, unroll)
    @assert (L = 3 * unroll) == length(u)
    τ½ = τ/2
    
    lorenz_unroll!(du, u, unroll)
    @turbo for i in 1:L
        dx[i] = si * du[i]
        u_tmp[i] = u[i] + τ½ * du[i]
    end

    lorenz_unroll!(du, u_tmp, unroll)
    @turbo for i in 1:L
        dx[i] = dx[i] + 2*si * du[i]
        u_tmp[i] = u[i] + τ½ * du[i]
    end

    lorenz_unroll!(du, u_tmp, unroll)
    @turbo for i in 1:L
        dx[i] = dx[i] + 2*si * du[i]
        u_tmp[i] = u[i] + τ * du[i]
    end

    lorenz_unroll!(du, u_tmp, unroll)
    @turbo for i in 1:L
        dx[i] = dx[i] + si * du[i]
        u_tmp[i] = u[i] + τ * dx[i]
    end

    return u_tmp
end

function lorenz_rk4_flow_map(u; unroll=1, step_size=0.01, steps=20)
    du, dx, u_tmp = similar(u), similar(u), similar(u)
    for _ = 1:steps
        u = lorenz_rk4_unroll(du, dx, u_tmp, u, step_size, unroll)
    end
    return u
end

function map_boxes(g::BoxMap, source::BoxSet; unroll=4)
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]
    L = 3 * unroll

    @inbounds @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = reduce(vcat, g.domain_points(c, r))
        N = length(points)
        k, rem = divrem(N, L)
        
        @fastmath @muladd @turbo for m in 1:3:N-2
            points[m]   = c[1] + r[1] * points[m]
            points[m+1] = c[2] + r[2] * points[m+1]
            points[m+2] = c[3] + r[3] * points[m+2]
        end

        if k != 0
            for i in 1:L:(k-1)*L+1#N-L+1
                fp = g.map(points[i:i+L-1]; unroll=unroll)

                for j in 1:3:L-2
                    hit = point_to_key(P, fp[j:j+2])

                    if !isnothing(hit)
                        push!(image[threadid()], hit)
                    end
                end
            end
        #else
            #@debug """Unroll factor too large for number of points. 
            #Defaulting to single-point loops.""" N L
        end 

        if rem != 0
            #@debug """Unroll complete, but points remain. 
            #Remainder of points are calulated in single-point-loops.""" N L rem
            for i in k*L+1:3:k*L+rem-2
                fp = g.map(points[i:i+2]; unroll=1)
                hit = point_to_key(P, fp)

                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end 



center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
F = BoxMap(lorenz_rk4_flow_map, P; no_of_points=16)

const x_eq = SVector{3, Float64}([sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1])
function S_init()
    points = [SVector{3, Float64}(x_eq .+ SVector{3, Float64}(15 .* rand(3))) for _ in 1:100]
    BoxSet(P, Set(point_to_key(P, point) for point in points))
end

S_mapped = map_boxes(F, S_init())
@benchmark map_boxes(F, S_init())
# mean of 900μs per run compared to 2.5ms in control run. That's a 2.5x speedup!