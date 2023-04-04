using GAIO
using Documenter, LinearAlgebra, SparseArrays, StaticArrays
import Plots, GLMakie

ENV["JULIA_DEBUG"] = Documenter
ENV["GKSwstype"] = "100"
ci = get(ENV, "CI", nothing) == "true"

makedocs(
    modules = [GAIO],
    sitename = "GAIO.jl",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "General usage" => "general.md",
        "BoxMaps" => [
            "boxmaps/boxmaps_general.md",
            "boxmaps/montecarlo.md",
            "boxmaps/grid.md",
            "boxmaps/adaptive.md",
            "boxmaps/interval.md",
            "boxmaps/boxmaps_simd.md",
            "boxmaps/boxmaps_cuda.md",
            "boxmaps/pointdiscretized.md",
            "boxmaps/sampled.md",
            "boxmaps/new_types.md"
        ],
        #"BoxMaps" => "boxmap.md",
        "Algorithms" => [
            "algorithms/relative_attractor.md",
            "algorithms/unstable_manifold.md",
            "algorithms/chain_recurrent_set.md",
            "algorithms/transfer_operator.md",
            "algorithms/root_covering.md",
            "algorithms/ftle.md",
            "algorithms/implicit_manifold.md",
            "algorithms/seba.md",
            "algorithms/box_dimension.md",
            "algorithms/almost_invariant_coherent_sets.md",
            "algorithms/entropy.md",
            "algorithms/pareto_set.md",
            "algorithms/control.md"
        ],
        #"Algorithms" => "algorithms.md",
        "Plotting" => "plotting.md",
        "Maximizing Performance" => [
            "simd.md",
            "cuda.md"
        ],
        "Other Examples" => "examples.md",
        "Library Reference" => "library_reference.md"
    ],
    doctest = false,
    format = Documenter.HTML(prettyurls = ci)
    #format = Documenter.LaTeX(platform = "none")
)

username = get(ENV, "GITHUB_REPOSITORY", nothing)

if !( username in ["gaioguys/GAIO.jl", "April-Hannah-Lena/GAIO.jl"] )
    username = "gaioguys/GAIO.jl"
end

if ci
    deploydocs(
        repo = "github.com/" * username * ".git",
        push_preview = true,
        versions = nothing
    )
end
