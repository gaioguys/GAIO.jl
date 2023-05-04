module GAIO

# general dependencies
using LinearAlgebra
using StaticArrays
using MuladdMacro
using PrecompileTools

# for mapping boxes
using FLoops
using OrderedCollections
using SplittablesBase
using SplitOrderedCollections
using IntervalArithmetic

# graph / linear algebra algorithms
using Graphs
using SparseArrays
using Arpack

# plotting using Makie -> in future versions maybe as Extensions
using GeometryBasics
using MakieCore
using MakieCore: @recipe

# misc
import Base: unsafe_trunc
import Base: ∘
import Base: @propagate_inbounds

export Box
export volume, center, vertices, rescale

export AbstractBoxPartition, BoxPartition, TreePartition
export key_to_box, point_to_key, bounded_point_to_key, point_to_box
export depth, tree_search

export BoxSet
export cover, subdivide, subdivide!

export BoxFun

export TransferOperator
export construct_transfers, eigs, svds

export BoxGraph, Graph
export union_strongly_connected_components

export BoxMap, map_boxes
export SampledBoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
export AdaptiveBoxMap, approx_lipschitz, sample_adaptive
export SmartBoxMap
export IntervalBoxMap

export rk4, rk4_flow_map
export preimage, symmetric_image

export relative_attractor, maximal_forward_invariant_set, maximal_invariant_set
export unstable_set, chain_recurrent_set
export box_dimension, finite_time_lyapunov_exponents
export armijo_rule, adaptive_newton_step, cover_roots, cover_manifold
export nth_iterate_jacobian
export seba, partition_unity, partition_disjoint, partition_likelihood

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
include("boxmap_intervals.jl")
include("boxmap.jl")

include("boxfun.jl")  
include("transfer_operator.jl")
include("boxgraph.jl")
include("algorithms.jl")

const default_box_color = :red # default color for plotting

include("makie.jl")

include("precompile.jl")

end # module
