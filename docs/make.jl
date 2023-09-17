using GAIO
using Documenter, LinearAlgebra, SparseArrays, StaticArrays, Graphs, MetaGraphsNext, SIMD
import Plots, MakieCore, GLMakie

ci = get(ENV, "CI", "false") == "true"

ENV["JULIA_DEBUG"] = Documenter #nothing
ENV["n_frames"] = ci ? 200 : 10
ENV["GKSwstype"] = "100"

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
        "Invariant Sets" => [
            "Relative Attractor" => "algorithms/relative_attractor.md",
            "Chain Reccurent Set" => "algorithms/chain_recurrent_set.md",
            "Maximal Invariant Set" => "algorithms/maximal_invariant_set.md",
            "Stable and Unstable Manifold" => "algorithms/unstable_manifold.md"
        ],
        "Transfer- and Koopman Operators" => [
            "Ulam's method and Invariant Measures" => "algorithms/transfer_operator.md",
            "Almost Invariant (metastable) Sets" => "algorithms/almost_invariant.md",
            "Cyclic Sets" => "algorithms/cyclic.md",
            "Coherent Sets" => "algorithms/coherent.md",
            "Extracting Multiple Sets via SEBA" => "algorithms/seba.md"
        ],
        "Scalar Diagnostics" => [
            "Fractal Dimension" => "algorithms/box_dimension.md",
            "Lyapunov Exponents / FTLEs" => "algorithms/ftle.md",
            #"Topological Entropy" => "algorithms/entropy.md"
        ],
        "Conley-Morse Theory" => [
            "Morse Graph" => "algorithms/morse_graph.md",
            "Conley Index" => "algorithms/conley_index.md"
        ],
        "Misceallenous Algorithms" => [
            "Root Covering" => "algorithms/root_covering.md",
            "Covering Implicitly Defined Manifolds" => "algorithms/implicit_manifold.md"
        ],
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

if ci
    username = get(ENV, "GITHUB_REPOSITORY", "gaioguys/GAIO.jl")
    deploydocs(
        repo = "github.com/" * username * ".git",
        push_preview = true,
        versions = nothing
    )
end
