# `CPUSampledBoxMap`

Naturally, if an increase in accuracy is desired in a `SampledBoxMap`, a larger set of test points may be chosen. This leads to a dilemma: the more accurate we wish our approximation to be, the more we need to map very similar test points forward, causing a considerable slow down for complicated dynamical systems. However, the process of mapping each test point forward is completely independent on other test points. This means we do not need to perform each calculation sequentially; we can parallelize. 

If the point map only uses "basic" instructions, then it is possible to simultaneously apply Single Instructions to Multiple Data (SIMD). This way multiple funnction calls can be made at the same time, increasing performance. For more details, see the [maximizing performance section](https://gaioguys.github.io/GAIO.jl/simd/). 

```@docs
GridBoxMap(c::Val{:simd}, map, domain::Box{N,T}; no_of_points) where {N,T}
MonteCarloBoxMap(c::Val{:simd}, map, domain::Box{N,T}; no_of_points) where {N,T}
```
