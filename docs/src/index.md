# GAIO.jl

## About

GAIO (_Global Analysis of Invariant Objects_) is a Julia package for set oriented computations.  Sets are represented by  collections of boxes. A _box_ (or _cube_) is a higher dimensional interval, i.e. a set of the form
```math
[a_1,b_1) × ... × [a_n,b_n),    a_k,b_k ∈ ℝ
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

## Project origin

The package originated as a university seminar for master students in the summer 2020, with the aim to enhance 
and convert the already existing Matlab GAIO package to Julia.
