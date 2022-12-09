# Usage

The base of the numerical set oriented methods of this framework are `BoxSet` (the discretization of a set) and `BoxMap` (the discretization of a map), thus, in the following, we will have a closer look at the two and other useful things to know when using GAIO.jl. 

To create a `Box` given its center point `c = (c_1, c_2, ..., c_d)` as well as its "radius" in every axis direction `r = (r_1, r_2, ..., c_d)`, simply type 
```julia
using GAIO

Q = Box(c, r)
```

This creates a set ``Q = [c_1 - r_1, c_1 + r_1), \times \ldots \times [c_d - r_d, c_d + r_d)``. Conversely, one can get back the vectors `c` and `r` by calling
```julia
c, r = Q
```

## BoxPartition

Most algorithms in GAIO.jl revolve around a partition of the domain ``Q`` into small boxes. To create an ``n_1 \times \ldots \times n_d`` - element equidistant grid of boxes, we can pass the tuple ``n = (n_1, \ldots, n_d)`` into the function `BoxPartition`
```julia
P = BoxPartition(Q, n)
```

`BoxPartition`s use a cartesian indexing structure to be memory-efficient. These indices are accessed and used through the API:
```julia
key = point_to_key(P, x)    # x is some point in the domain Q

box = key_to_box(P, key)    # cover the point x with a box from P

box = point_to_box(P, x)    # performs both above functions
```

## TreePartition

For partitions of ``Q`` into variably sized boxes, one can use `TreePartition`:
```julia
P = TreePartition(Q)
```

!!! warning "Using `TreePartition`"
    `TreePartition` is an area of active development, and an overhaul is potentially planned in the future. Please do not use `TreePartition` yet. 

## BoxSet

The core idea behind GAIO.jl is to approximate an subset of the domain via a collection of small boxes. To construct `BoxSet`s, there are two main options: getting all boxes in the partition, or locating a box surrounding a point ``x \in Q``
```julia
B = P[:]    # set of all boxes in P

B = P[x]    # one box surrounding the point x
```

One can also create a `Boxset` from an iterable of `Box`es. This will cover every element of the iterable with boxes from `P`:
```julia
S = [Box(center_1, radius_1), Box(center_2, radius_2), Box(center_3, radius_3)] # etc... 

B = P[S]
```

`BoxSet` is a highly memory-efficient way of storing boxes. However, should you want to access the boxes or their internal data, this can be done via iteration:
```julia
for box in B
    center, radius = box
    # do something
end

# get an array of boxes
arr_of_boxes = collect(B)

# get an array of box centers
arr_of_centers = collect(box.center for box in B)

# get an array of box radii
arr_of_radii = collect(box.radius for box in B)

# (memory-efficiently) create a matrix where each center is a column
mat_of_centers = reinterpret(reshape, eltype(arr_of_centers[1]), arr_of_centers)
```

## BoxMap

A BoxMap is a function which maps boxes to boxes. Given a pointmap `f`, initialize the corresponding `BoxMap` `F` by
```julia
F = BoxMap(f, P.domain)
```
This will generate a `BoxMap` which uses Monte-Carlo test points to map boxes. To specify the amount of test points used, use the `no_of_points` keyword argument:
```julia
F = BoxMap(f, P.domain, no_of_points=300)
```

## AdaptiveBoxMap

For choosing test points we can use some knowledge of the Lipschitz matrix for ``f`` in a box `Box(c, r)`, that is, a matrix ``L \in \mathbb{R}^{d \times d}`` such that 
```math
| f(y) - f(z) | \leq L \, | y - z | \quad \text{for all } y, z \in \text{Box}(c, r),
```
where the operations ``| \cdot |`` and `` \leq `` are to be understood elementwise. The function `AdaptiveBoxMap` attempts to approximate ``L`` before choosing an adaptive grid of test points in each box, as described in [1]
```julia
F = AdaptiveBoxMap(f, P.domain)
```

## Using BoxMap / AdaptiveBoxMap

Now, one can map a `BoxSet` via the `BoxMap` `F` by simply calling `F` as a function 
```julia
C = F(B)
```
where the output `C` is also a `BoxSet`.

## TransferOperator

The _Perron-Frobenius operator_ (or _transfer operator_) [2] is discretized in GAIO.jl using the `TransferOperator` type. To initialize a `TransferOperator` that acts on a subdomain of ``Q``, type
```julia
T = TransferOperator(F, B)   # T operates on the domain covered by the box set B
```
To find an approximate invariant measure over `B` use the `eigs` function
```julia
λ, ev, num_converged_eigs = eigs(T)

μ = ev[1]   # ev is an array of measures, grab the first one
```
This can also be done with the adjoint _Koopman operator_ `T'`. 

## BoxFun

The return type of `eigs(T)` is a stepwise constant function over the boxes in `B`, which is called a `BoxFun`. One can let `T` act on a `BoxFun` simply through multiplication
```julia
ν = T * μ
```
Of course, the same holds for the the Koopman operator as well. 

## BoxGraph

One could equivalently view the transfer operator as a weighted directed graph. That is, a transfer operator in GAIO.jl is the (transposed) weighted adjacency matrix for a graph. This graph can be constructed using
```julia
G = Graph(T)    # T is a transfer operator
```
The return type is a `BoxGraph`. `Boxgraph` is hooked into the `Graphs.jl` interface, which means all algorithms or etc. from Graphs.jl should work "out of the box". To construct a BoxSet from some index / indices of vertices in a BoxGraph, call
```julia
BoxSet(G, vertex_index_or_indices)
``` 
See the docstring for `BoxGraph` for details on how to translate between GAIO.jl and Graphs.jl. 

## Plotting

GAIO.jl offers both `Plots` or `Makie` for plotting. To plot a `BoxSet` or a `BoxFun`, simply choose either Plots or a Makie backend, eg. `GLMakie`, and call `plot` on a `BoxSet` or `BoxFun`
```julia
using GLMakie: plot

plot(B)
```
Plotting works with all the functionality of either package. This means you can set box plots as subplots, add colorbars, etc., using the Plots or Makie interface. For an example, see `examples/invariant_measure_2d.jl`. 

## References

[1] Oliver Junge. “Rigorous discretization of subdivision techniques”. In: _International Conference on Differential Equations_. Ed. by B. Fiedler, K. Gröger, and J. Sprekels. 1999.

[2] Andrzej Lasota and Michael C. Mackey. _Chaos, Fractals, and Noise. Stochastic Aspects of Dynamics_. Springer New York, NY, 1994. doi: https://doi.org/10.1007/978-1-4612-4286-4.
