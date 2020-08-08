module GAIO

using LinearAlgebra
using SparseArrays
using StaticArrays
using GLFW
using ModernGL
using LightGraphs
using Arpack

export Box

export BoxPartition, RegularPartition, TreePartition
export dimension, depth

export BoxSet
export boxset_empty, subdivide, subdivide!

export BoxFun

export TransferOperator
export strongly_connected_components, matrix, eigs

export BoxMap, PointDiscretizedMap
export boxmap

export relative_attractor, unstable_set!, chain_recurrent_set, cover_roots

export plot

include("box.jl")

abstract type BoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxfun.jl")
include("transfer_operator.jl")
include("boxmap.jl")
include("algorithms.jl")

# visualization
include("plot/shader.jl")
include("plot/camera.jl")
include("plot/plot.jl")

end # module
