using GAIO
using Documenter, LinearAlgebra, SparseArrays

makedocs(
    modules = [GAIO],
    sitename = "GAIO.jl",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "General usage" => "general.md",
        "Algorithms" => "algorithms.md",
        "BoxMaps" => "boxmap.md",
        "Maximizing Performance" => [
            "simd.md",
            "cuda.md"
        ],
        "Plotting" => "plotting.md",
        "Examples" => "examples.md",
        "Reference" => [
            "data_structures.md",
            "library_reference.md"
        ]
    ],
    doctest = false
)

deploydocs(
    repo = "github.com/gaioguys/GAIO.jl.git",
    push_preview = true,
    versions = nothing
)
