module GAIO

# general dependencies
using LinearAlgebra
using StaticArrays
using MuladdMacro
using PrecompileTools
using TupleTools
const tuple_deleteat = TupleTools.deleteat

# for mapping boxes
using FLoops
using OrderedCollections
using SplittablesBase
using SplitOrderedCollections
using IntervalArithmetic

# graph / linear algebra algorithms
#using Graphs
using MatrixNetworks
using MatrixNetworks: Strong_components_output, Strong_components_rich_output
using SparseArrays
using Arpack

# misc
import Base: unsafe_trunc, IteratorEltype, HasEltype, OneTo
import Base: ∘
import Base: @propagate_inbounds

export Box
export volume, center, radius, vertices, rescale

export AbstractBoxPartition, BoxPartition, TreePartition
export key_to_box, point_to_key, bounded_point_to_key, point_to_box
export depth, tree_search, find_at_depth, leaves, hidden_keys

export BoxSet
export cover, subdivide, subdivide!, neighborhood, nbhd, marginal, density

export BoxMeasure

export TransferOperator
export construct_transfers, eigs, svds
export key_to_index, index_to_key

#export BoxGraph, Graph
#export union_strongly_connected_components

export BoxMap, map_boxes
export SampledBoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
export AdaptiveBoxMap, approx_lipschitz, sample_adaptive
export SmartBoxMap
export IntervalBoxMap

export rk4, rk4_flow_map
export preimage, symmetric_image
export morse_map, morse_component_map, morse_graph, morse_tiles, morse_graph_and_tiles
export isolating_neighborhood, isloating_nbhd, index_pair, index_quad, @save

export relative_attractor, maximal_forward_invariant_set, maximal_backward_invariant_set, maximal_invariant_set
export unstable_set, chain_recurrent_set
export box_dimension, finite_time_lyapunov_exponents
export armijo_rule, adaptive_newton_step, cover_roots, cover_manifold
export nth_iterate_jacobian
export seba, partition_unity, partition_disjoint, partition_likelihood

# ENV["JULIA_DEBUG"] = all

const SVNT{N,T} = Union{<:NTuple{N,T}, <:StaticVector{N,T}}
const default_box_color = :red # default color for plotting

# we need a small helper function because of 
# how julia dispatches on `union!`
⊔(set1::AbstractSet, set2::AbstractSet) = union!(set1, set2)
⊔(set1::AbstractSet, object) = union!(set1, (object,))
⊔(set1::AbstractSet, ::Nothing) = set1

⊔(d::AbstractDict...) = mergewith!(+, d...)
⊔(d::AbstractDict, p::Pair...) = foreach(q -> d ⊔ q, p)
⊔(d::AbstractDict, ::Nothing) = d
⊔(d::AbstractDict, ::Pair{<:Tuple{Nothing,<:Any},<:Any}) = d

function ⊔(d::AbstractDict, p::Pair)
    k, v = p
    d[k] = haskey(d, k) ? d[k] + v : v
    d
end

include("box.jl")

abstract type AbstractBoxPartition{B <: Box} end

include("partition_regular.jl")
include("partition_tree.jl")
include("boxset.jl")

abstract type BoxMap end

include("boxmap_intervals.jl")
include("boxmap_sampled.jl")
include("boxmap.jl")

include("boxmeasure.jl")  
include("transfer_operator.jl")
#include("boxgraph.jl")

Base.@deprecate_binding BoxFun BoxMeasure

include("algorithms/invariant_sets.jl")
include("algorithms/scalar_diagnostics.jl")
include("algorithms/optimization.jl")
include("algorithms/seba.jl")
include("algorithms/morse_graph.jl")
include("algorithms/conley_index.jl")
include("algorithms/maps.jl")

const nbhd = neighborhood

include("precompile.jl")

end # module
