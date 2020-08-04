module GAIO

using LinearAlgebra
using StaticArrays
using GLFW
using ModernGL
using LightGraphs

export Box, BoxSet
export boxset_empty, subdivide, subdivide!

export BoxPartition, RegularPartition, TreePartition
export dimension, depth

export BoxMap, PointDiscretizedMap
export boxmap

export relative_attractor, unstable_set!, chain_recurrent_set, cover_roots

export plot

include("box.jl")

"""
    BoxPartition{B <: Box}

Represents a partition of a given domain box of type `B` into several boxes of type `B`.
"""
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
