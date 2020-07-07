module GAIO

using StaticArrays

export Box, BoxSet
export boxset_empty, subdivide, subdivide!

export BoxPartition, RegularPartition, TreePartition
export dimension, depth

export BoxMap, PointDiscretizedMap
export boxmap

export relative_attractor, unstable_set!, chain_recurrent_set

include("box.jl")

abstract type BoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxmap.jl")
include("algorithms.jl")

# examples. TODO: move away from here
using QuadGK
using Interpolations
using LinearAlgebra
using LightGraphs

include("../examples/henon.jl")
include("../examples/lorenz.jl")
include("../examples/knotted_flow.jl")

end # module
