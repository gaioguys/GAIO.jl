module GAIO

using LinearAlgebra
using SparseArrays
using StaticArrays
using LightGraphs
using ForwardDiff
using Arpack
using Base.Threads

using GLFW
using ModernGL
using GeometryBasics
using GLMakie

export Box, volume

export AbstractBoxPartition, BoxPartition, TreePartition
export dimension, depth

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

export relative_attractor, unstable_set!, chain_recurrent_set, cover_roots, finite_time_lyapunov_exponents

export plot

include("box.jl")

abstract type AbstractBoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxmap.jl")
include("boxfun.jl")  
include("transfer_operator.jl")
include("algorithms.jl")
include("plot.jl")

end # module
