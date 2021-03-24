module GAIO

using LinearAlgebra
using StaticArrays
using GLFW
using ModernGL
using LightGraphs
using Base.Threads

export Box, BoxSet
export boxset_empty, subdivide, subdivide!

export BoxPartition, RegularPartition, TreePartition
export dimension, depth

export BoxMap, PointDiscretizedMap
export boxmap, map_boxes, map_boxes_new

export relative_attractor, unstable_set!, chain_recurrent_set, cover_roots

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
