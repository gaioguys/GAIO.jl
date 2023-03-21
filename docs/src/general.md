# Usage

The base of the numerical set oriented methods of this framework are `BoxSet` (the discretization of a set of boxes) and `BoxMap` (the discretization of a map). Thus, in the following, we will have a closer look at the two and other useful things to know when using GAIO.jl. 

To create a `Box` given its center point `c = (c_1, c_2, ..., c_d)` as well as its "radius" in every axis direction `r = (r_1, r_2, ..., c_d)`, simply type 
```@repl 1
using GAIO
c, r = (0.5, 0.5), (0.5, 0.5)
Q = Box(c, r)
```

This creates a set ``Q = [c_1 - r_1, c_1 + r_1), \times \ldots \times [c_d - r_d, c_d + r_d)``. Conversely, one can get back the vectors `c` and `r` by calling
```@repl 1
c, r = Q
```

## BoxPartition

Most algorithms in GAIO.jl revolve around a partition of the domain ``Q`` into small boxes. To create an ``n_1 \times \ldots \times n_d`` - element equidistant grid of boxes, we can pass the tuple ``n = (n_1, \ldots, n_d)`` into the function `BoxPartition`
```@repl 1
n = (4, 2)
P = BoxPartition(Q, n)
```

`BoxPartition`s use a cartesian indexing structure to be memory-efficient. These indices are accessed and used through the API:
```@repl 1
x = (0.2, 0.1)
key = point_to_key(P, x)    # x is some point in the domain Q
box = key_to_box(P, key)    # cover the point x with a box from P
box = point_to_box(P, x)    # performs both above functions
```

## TreePartition

For partitions of ``Q`` into variably sized boxes, one can use `TreePartition`:
```@repl 1
P2 = TreePartition(Q)
```
A `TreePartition` is a binary tree structure partitioning the domain. It can be split in half using the command 
```@repl 1
subdivide!(P2)
subdivide!(P2)
subdivide!(P2)
```
The axis direction along which to subdivide cycles through with the depth, i.e. subdividing at depth 1 splits along dimension 1, subdividing at depth `d+1` splits along dimension 1 again. 

The `TreePartition` created above is equivalent to a 4x2 `BoxPartition`. One can retrieve this using 
```@repl 1
P3 = BoxPartition(P2)
```
`TreePartition`s use indices of the type `(depth, cartesian_index)` where `cartesian_index` is the equivalent index of a `BoxPartition` with the same size as a `TreePartition` subdivided `depth` times. In other words,
```@repl 1
key_to_box( P, (1, 1) ) == key_to_box( P2, (4, (1, 1)) )
key_to_box( P, (4, 2) ) == key_to_box( P2, (4, (4, 2)) )
```

## BoxSet

The core idea behind GAIO.jl is to approximate an subset of the domain via a collection of small boxes. To construct `BoxSet`s, there are two main options: getting all boxes in the partition, or locating a box surrounding a point ``x \in Q``
```@repl 1
B = cover(P, x)    # one box surrounding the point x
B = cover(P, :)    # set of all boxes in P
```

One can also create a `Boxset` from an iterable of `Box`es. This will cover every element of the iterable with boxes from `P`:
```@repl 1
x1 = (0.2, 0.1)
box1 = point_to_box(P, x1)
x2 = (0.3, 0.6)
box2 = point_to_box(P, x2)
B = cover(P, [box1, box2])
```

`BoxSet` is a highly memory-efficient way of storing boxes. However, should you want to access the boxes or their internal data, this can be done via iteration:
```@repl 1
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

A BoxMap is a function which maps boxes to boxes. Given a pointmap `f`, 
```@repl 1
function f(u)   # the Baker transformation
    x, y = u
    if x < 0.5
        (2x, y/2)
    else
        (2x - 1, y/2 + 1/2)
    end
end
```
initialize the corresponding `BoxMap` `F` by
```@repl 1
F = BoxMap(f, Q)
```
This will generate a `BoxMap` which attempts to calculate setwise images of `f`. There are many types of `BoxMap` discretizations available, see the section on BoxMaps for more information. 

## Using BoxMap

Now, one can map a `BoxSet` via the `BoxMap` `F` by simply calling `F` as a function 
```@repl 1
C = F(B)
```
where the output `C` is also a `BoxSet`.

## TransferOperator

The _Perron-Frobenius operator_ (or _transfer operator_) [2] is discretized in GAIO.jl using the `TransferOperator` type. To initialize a `TransferOperator` that acts on a subdomain of ``Q``, type
```@repl 1
B = cover(P, :)
T = TransferOperator(F, B)   # T operates on the domain covered by the box set B
```
In this case, the codomain is generated automatically. This is not always ideal, so the codomain can be specified as an argument
```@repl 1
T = TransferOperator(F, B, B)
```
To convert this to the underlying transfer matrix described in [3], one cal simply type 
```@repl 1
Matrix(T)
```
To find an approximate invariant measure over `B` use the `eigs` function from `Arpack.jl`. All keyword arguments from `Arpack.eigs` are supported. 
```@repl 1
# for the Baker trafo, the Lebesque measure 
# - i.e. the constant-weight measure - is invariant
λ, ev = eigs(T);
λ
ev
μ = ev[1]   # ev is an array of measures, grab the first one
```
This can also be done with the adjoint _Koopman operator_ `T'`. 

## BoxFun

The return type of `eigs(T)` is a discretization of a measure over the domain. Specifically, it is a stepwise constant function defined on the boxes in `B`, which is called a `BoxFun`. One can let `T` act on a `BoxFun` simply through multiplication
```@repl 1
ν = T*μ
```
Of course, the same holds for the the Koopman operator as well. 
```@repl 1
ν = T'μ
```
Since a measure ``\mu`` is a function defined over measurable sets, composite measures ``g \circ \mu`` are well-defined for functions ``g : \mathbb{R} \to \mathbb{R}`` (or ``g : \mathbb{C} \to \mathbb{C}``). This is supported in GAIO.jl for `BoxFuns`
```@repl 1
η = exp ∘ μ
```
Similarly probability measures can be given a vector space structure. This is also supported in GAIO.jl
```@repl 1
ν + μ
2ν - μ/2
```

## BoxGraph

One could equivalently view the transfer operator as a weighted directed graph. That is, a transfer matrix in GAIO.jl is the (transposed) weighted adjacency matrix for a graph. This graph can be constructed using
```@repl 1
G = Graph(T)
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

[3] Michael Dellnitz, Oliver Junge, and Gary Froyland. “The Algorithms Behind GAIO - Set Oriented Numerical Methods for Dynamical Systems”. In: _Ergodic Theory, Analysis, and Efficient Simulations of Dynamical Systems_. Ed. by Bernold Fiedler. Springer Berlin, 2001, pp. 145–174. doi: https://doi.org/10.1007/3-540-35593-6.
