# GAIO.jl

Currently under development -- breaking changes may occur at any time ...

## About 

GAIO.jl is a Julia package for set oriented computations. Sets are represented by collections of boxes (i.e. cubes). GAIO.jl provides algorithms for 
* dynamical systems
  * invariant sets (maximal invariant set, chain recurrent set, (relative) attractor, (un-)stable manifold)
  * almost invariant and coherent sets
  * finite time Lyapunov exponents
  * entropy and box dimension
* root finding problems
* multi-objective optimization problems
* computing implicitly defined manifolds

## Documentation

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://gaioguys.github.io/GAIO.jl/)

## Installation

The package requires Julia 1.10 or later. In Julia's package manager, type
```julia
pkg> add https://github.com/gaioguys/GAIO.jl.git
```

## Getting started

The following script computes the attractor of the Hénon map within the box [-3,3)²: 

```julia
using GAIO
using Plots: plot

center, radius = (0,0), (3,3)
Q = Box(center, radius)                       # domain for the computation
P = BoxGrid(Q, (2, 2))                   # 2 x 2 partition of Q
S = cover(P, :)                               # Set of all boxes in P

f((x,y)) = (1 - 1.4*x^2 + y, 0.3*x)           # the Hénon map
F = BoxMap(f, P)                              # ... turned into a map on boxes

R = relative_attractor(F, S, steps = 18)      # subdivison algorithm computing
                                              # the attractor relative to Q
plot(R)                                       # plot R
```
![GitHub Logo](docs/src/assets/henon.svg)

For more examples, see the `examples\` folder.

## License

See `LICENSE` for GAIO.jl's licensing information.

## References

* Dellnitz, M.; Froyland, G.; Junge, O.: The algorithms behind GAIO - Set oriented numerical methods for dynamical systems, in: B. Fiedler (ed.): Ergodic theory, analysis, and efficient simulation of dynamical systems, Springer, 2001.
* Dellnitz, M.; Junge, O.: On the approximation of complicated dynamical behavior, SIAM Journal on Numerical Analysis, 36 (2), 1999.
* Dellnitz, M.; Hohmann, A.: A subdivision algorithm for the computation of unstable manifolds and global attractors. Numerische Mathematik 75, pp. 293-317, 1997.
