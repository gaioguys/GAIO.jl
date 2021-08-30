module GAIO

using LinearAlgebra
using StaticArrays
using GLFW
using ModernGL
using LightGraphs
using ForwardDiff
using Arpack
using Base.Threads

export Box

export BoxPartition, RegularPartition, TreePartition
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

export relative_attractor, unstable_set!, chain_recurrent_set, cover_roots, finite_time_lyapunov_exponents

export plot

include("box.jl")

abstract type BoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxmap.jl")
include("algorithms.jl")

# visualization
include("plot/shader.jl")
include("plot/camera.jl")
include("plot/plot.jl")

end # module
