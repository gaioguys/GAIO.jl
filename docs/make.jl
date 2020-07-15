using GAIO
using Documenter

makedocs(
    modules = [GAIO],
    sitename = "GAIO.jl",
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/cafaxo/GAIO.jl.git",
    devbranch = "actions"
)
