using GAIO
using Documenter

makedocs(
    modules = [GAIO],
    sitename = "GAIO.jl",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "General usage" => "general.md",
        "Algorithms" => "algorithms.md",
        "Plotting" => "plotting.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(
    repo = "github.com/gaioguys/GAIO.jl.git",
    push_preview = true
)
