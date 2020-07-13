using GAIO
using Documenter

makedocs(
    modules = [GAIO],
    sitename = "GAIO.jl",
    pages = [
        "Home" => "index.md",
        "General usage" => "general.md",
        "Algorithms" => "algorithms.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(
    repo = "github.com/gaioguys/GAIO.jl.git",
    devbranch = "initialdocs2"
)
