using GAIO
using BenchmarkTools

SUITE = BenchmarkGroup(["GAIO"])

# TODO: include benchmark files here
include("samplebench.jl")
