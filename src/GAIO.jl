module GAIO

using LinearAlgebra
using SparseArrays
using StaticArrays
# using GLFW
# using ModernGL
using GeometryBasics
using Graphs
using ForwardDiff
using Arpack
using Base.Threads
using Base: unsafe_trunc
using MuladdMacro
using HostCPUFeatures
using SIMD
using Adapt
using CUDA
using Base.Iterators: Stateful, take

using GLMakie
using WGLMakie

export Box

export AbstractBoxPartition, BoxPartition, TreePartition
export depth, key_to_box, point_to_key, tree_search

export BoxSet
export boxset_empty, subdivide, subdivide!

export BoxFun

export TransferOperator
export strongly_connected_components, matrix, eigs

export BoxMap, PointDiscretizedMap
export SampledBoxMap, AdaptiveBoxMap
export boxmap

export map_boxes, map_boxes_new

export rk4, rk4_flow_map

export relative_attractor, unstable_set!, chain_recurrent_set
export cover_roots, finite_time_lyapunov_exponents

export plot

# ENV["JULIA_DEBUG"] = all

include("box.jl")

abstract type AbstractBoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxmap.jl")
include("boxmap_simd.jl")
include("boxfun.jl")  
include("transfer_operator.jl")
include("algorithms.jl")
include("plot.jl")

if CUDA.functional()
    print("CUDA acceleration available")
    include("boxmap_cuda.jl")    
end

end # module
