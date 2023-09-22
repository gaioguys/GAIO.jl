# runs from GAIO.jl/examples/Project.toml
using Pkg
scratchspace = pwd() * "/test/logs"

redirect_stdio(stderr="$scratchspace/std.err") do
    Pkg.test("GAIO")
end
