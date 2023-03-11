# `GPUSampledBoxmap`

If an Nvidia gpu is available, the above technique can be improved dramatically. The gpu uses a "massively parallel programming" paradigm, which fits perfectly to the problem of mapping many sample points independently. For more information, see the [maximizing performance section](https://gaioguys.github.io/GAIO.jl/cuda/).

```@docs
GridBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points) where {N,T}
MonteCarloBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points) where {N,T}
```
