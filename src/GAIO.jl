module GAIO

using StaticArrays

export Box, BoxSet
export boxset_empty, boxset_full, subdivide, subdivide!

export BoxPartition, RegularPartition, TreePartition
export dimension, depth

export BoxMap, PointDiscretizedMap
export boxmap

export relative_attractor, unstable_set!

include("box.jl")

abstract type BoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")
include("boxmap.jl")
include("algorithms.jl")

# examples. TODO: move away from here
include("../examples/henon.jl")
include("../examples/lorenz.jl")

end # module
