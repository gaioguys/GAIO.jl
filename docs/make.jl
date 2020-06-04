using GAIO
using Documenter

makedocs(;
    modules=[GAIO],
    authors="The GAIO.jl Team",
    repo="https://gitlab.lrz.de/software1/GAIO.jl/blob/{commit}{path}#L{line}",
    sitename="GAIO.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://software1.pages.gitlab.lrz.de/GAIO.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
