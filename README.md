# GAIO.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://gaioguys.github.io/GAIO.jl/dev/)

## About 

GAIO.jl is a Julia package for set oriented computations.  Sets are represented by box collections. A _box_ (or _cube_) is a higher dimensional interval, i.e. a set of the form
```math
[a₁,b₁) × ... × [aₙ,bₙ),    aₖ,bₖ ∈ ℝ
```
GAIO.jl provides algorithms for  
* dynamical systems
  * invariant sets (maximal invariant set, chain recurrent set, (relative) attractor, (un-)stable manifold)
  * almost invariant and coherent sets
  * finite time Lyapunov exponents
  * entropy and box dimension
* root finding problems
* multi-objective optimization problems
* computing implicitely defined manifolds


## Installation

The package requires Julia 1.7 or later. In Julia's package manager, type
```julia
pkg> add https://github.com/gaioguys/GAIO.jl.git
```
followed by
```julia
julia> using GAIO
```
at the Julia prompt in order to load the package.

## Documentation

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://gaioguys.github.io/GAIO.jl/dev/)


## Getting started

The following script computes the chain recurrent set of the Hénon map within the box [-3,3]²: 

```julia
using GAIO

center, radius = (0,0), (3,3)
Q = Box(center, radius)                       # domain for the computation
P = BoxPartition(Q)                           # 1 x 1 partition of Q

f((x,y)) = (1 - 1.2*x^2 + y, 0.3*x)           # the Hénon map
F = BoxMap(f, P)                              # ... turned into a map on boxes
R = chain_recurrent_set(F, P[:], steps = 15)  # subdivison algorithm computing
                                              # the chain recurrent set R in Q
plot(R)                                       # plot R
```
![GitHub Logo](henon.svg)

For more examples, see the `examples\` folder.

## License

See `LICENSE` for GAIO.jl's licensing information.

