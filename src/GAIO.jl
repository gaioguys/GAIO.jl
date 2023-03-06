module GAIO

using LinearAlgebra
using SparseArrays
using StaticArrays
using OrderedCollections
# using GLFW
# using ModernGL
using GeometryBasics
using Graphs
using Arpack
using Base.Threads
using FLoops
using SplittablesBase
using SplitOrderedCollections
using Base: unsafe_trunc
using IntervalArithmetic
using MuladdMacro
using HostCPUFeatures
using SIMD
using Adapt
using CUDA
using Base.Iterators: Stateful, take

# using GLMakie
# using WGLMakie
using MakieCore
using MakieCore: @recipe
using RecipesBase: RecipesBase

import Base: ∘

export Box
export volume, center, vertices, rescale

export AbstractBoxPartition, BoxPartition, TreePartition
export key_to_box, point_to_key, bounded_point_to_key, point_to_box
export depth, tree_search

export BoxSet
export boxset_empty, subdivide, subdivide!

export BoxFun

export TransferOperator
export construct_transfers, eigs

export BoxGraph

export BoxMap, map_boxes
export SampledBoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
export AdaptiveBoxMap, approx_lipschitz, sample_adaptive
export CPUSampledBoxMap, GPUSampledBoxMap
export SmartBoxMap
export IntervalBoxMap

export rk4, rk4_flow_map

export relative_attractor, unstable_set, chain_recurrent_set
export cover_roots, finite_time_lyapunov_exponents
export SEBA

export plotboxes, plotboxes!

# ENV["JULIA_DEBUG"] = all

const SVNT{N,T} = Union{<:NTuple{N,T}, <:StaticVector{N,T}}

include("box.jl")

abstract type AbstractBoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")

abstract type BoxMap end

(g::BoxMap)(source::BoxSet) = map_boxes(g, source)

include("boxmap_sampled.jl")
include("boxmap_simd.jl")
include("boxmap_intervals.jl")
include("boxmap.jl")

include("boxfun.jl")  
include("transfer_operator.jl")
include("boxgraph.jl")
include("algorithms.jl")
include("plot.jl")

if CUDA.functional()
    include("boxmap_cuda.jl")
else
    include("no_boxmap_cuda.jl")
end

end # module
