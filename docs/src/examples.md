# Examples

# How to Use

To see examples of how to use GAIO.jl explore the [examples folder](https://github.com/gaioguys/GAIO.jl/tree/master/examples). This contains a `Project.toml` file that can be used to download all the necessary dependencies for the examples. 

To try the examples, one can choose between two options:
* (recommended) simply download an example file and see what packages are used, add them in the package manager
* clone the [GAIO.jl GitHub repository](https://github.com/gaioguys/GAIO.jl.git). Navigate to the repository folder `/path/to/GAIO.jl/examples` in the julia REPL using `pwd` and `cd`. Finally, call the following in the julia REPL
```julia
pkg> activate .
julia> pwd()
"/path/to/GAIO.jl"

julia> cd("./examples")

(GAIO) pkg> activate .
  Activating project at `/path/to/GAIO.jl/examples`

(examples) pkg> dev ../
   Resolving package versions...
  No Changes to `/path/to/GAIO.jl/examples/Project.toml`
  No Changes to `/path/to/GAIO.jl/examples/Manifest.toml`

(examples) pkg> instantiate
  69 dependencies successfully precompiled in 74 seconds
```

## Example Files

* [Relative attractor of the Hénon map](https://github.com/gaioguys/GAIO.jl/blob/master/examples/attractor.jl)
* [Unstable manifold of the Lorenz system](https://github.com/gaioguys/GAIO.jl/blob/master/examples/unstable_manifold.jl)
* [Recurrent of a knotted flow](https://github.com/gaioguys/GAIO.jl/blob/master/examples/recurrent_set.jl)
* [Invariant measure of the transfer operator for the logistic map](https://github.com/gaioguys/GAIO.jl/blob/master/examples/invariant_measure_1d.jl)
* [Invariant measure of the transfer operator for the Hénon map over the relative attractor](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/invariant_measure_2d.jl)
* [Invariant measure of the transfer operator for the Lorenz map over the unstable manifold](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/invariant_measure_3d.jl)
* [Almost invariant sets in Chua's circuit](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/almost_invariant_sets.jl)
* [Root covering of a cubic map](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/roots.jl)
* [Doubling performance using SIMD operations](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/fast_maps_using_SIMD.jl)
* [Vastly improving performance using Nvidia CUDA](https://github.com/gaioguys/GAIO.jl/blob/master/examples/advanced/fast_maps_using_CUDA.jl)
