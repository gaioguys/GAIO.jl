# GAIO.jl

GAIO (_Global Analysis of Invariant Objects_) is a Julia package for set oriented computations.  Sets are represented by box collections. A _box_ (or _cube_) is a higher dimensional interval, i.e. a set of the form
```math
[a₁,b₁] × ... × [aₙ,bₙ],    aₖ,bₖ ∈ ℝ
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

### Project origin

The package originated as a university seminar for master students in the summer 2020, with the aim to enhance and convert the already existing Matlab GAIO package to Julia.
