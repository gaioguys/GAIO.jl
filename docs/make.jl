using Documenter, DocumenterCitations#, JSServe
using Base: get_extension

using GAIO
using MetaGraphsNext
import Plots, GLMakie#, WGLMakie
import GraphRecipes, GraphMakie
import DynamicalSystems, OrdinaryDiffEq
import SIMD, CUDA
using LinearAlgebra, SparseArrays, StaticArrays, Graphs, Arpack, Serialization

const center = GAIO.center
const radius = GAIO.radius
const vertices = GAIO.vertices
const neighborhood = GAIO.neighborhood

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"))
ci = get(ENV, "CI", "false") == "true"

#ENV["JULIA_DEBUG"] = Documenter
ENV["JULIA_DEBUG"] = nothing

ENV["n_frames"] = ci ? 120 : 20
ENV["GKSwstype"] = "100"

pages = [
    "Home" => "index.md",
    #"Testing Page" => "test.md",
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
        "Reccurent Set" => "algorithms/recurrent_set.md",
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
        "cuda.md",
        "metal.md"
    ],
    "Other Examples" => "examples.md",
    "Data Structures" => "data_structures.md", 
    "Library Reference" => "library_reference.md",
    "References" => "references.md"
]

makedocs(
    modules = [
        GAIO,
        get_extension(GAIO, :MetaGraphsNextExt),
        get_extension(GAIO, :SIMDExt),
        get_extension(GAIO, :CUDAExt),
        get_extension(GAIO, :PlotsExt),
        get_extension(GAIO, :MakieExt)
    ],
    sitename = "GAIO.jl",
    pages = pages,
    pagesonly = true,
    doctest = false,
    draft = false,
    warnonly = !ci,
    #format = Documenter.LaTeX(platform = "none"),
    format = Documenter.HTML(
        prettyurls = ci, 
        size_threshold = 11981529, 
        assets = String["assets/citations.css"]
    ),
    plugins = [bib]
)

if ci
    username = get(ENV, "GITHUB_REPOSITORY", "gaioguys/GAIO.jl")
    deploydocs(
        repo = "github.com/" * username * ".git",
        push_preview = true,
        versions = nothing
    )
end
