# Usage

The base of the set oriented framework are `BoxSet`s (as a discretization of state space) and `BoxMap`s (as a discretization of a map). In the following, we will have a closer look at the two concepts and other useful things to know when using GAIO.jl. 

## Box

To create a `Box` with center `c`$\in\mathbb{R}^d$ and radius `r`$\in\mathbb{R}^d$, type 
```@repl 1
using GAIO
c, r = (0.5, 0.5), (0.5, 0.5)
box = Box(c, r)
```

This creates the set $box = [c_1 - r_1, c_1 + r_1) \times \ldots \times [c_d - r_d, c_d + r_d)$. Conversely, one can get back the vectors `c` and `r` from a box ``box`` by 
```@repl 1
c, r = box
```

## BoxGrid

Most algorithms in GAIO.jl revolve around a partition $\scr P$  of some domain $X\subset\mathbb{R}^d$ into small boxes.  Often, the domain `X` is a `Box` and the partition `𝒫` is a grid of boxes on `X`. To create an $n_1 \times \ldots \times n_d$-element grid of boxes on a box `X`, pass `X` and the tuple `n = (n_1,...,n_d)` to the function `BoxGrid`
```@repl 1
n = (4, 2)
𝒫 = BoxGrid(X, n)
```

<!--`BoxGrid`s use a cartesian indexing structure to be memory-efficient. These indices are accessed and used through the API:
```@repl 1
x = (0.2, 0.1)
key = point_to_key(P, x)    # x is some point in the domain Q
box = key_to_box(P, key)    # cover the point x with a box from P
box = point_to_box(P, x)    # performs both above functions
```-->

## BoxTree

For partitions of `X` into variably sized boxes, one can use `BoxTree`:
```@repl 1
𝒫 = BoxTree(X)
```
A `BoxTree` uses a binary tree structure to store a box partition of the domain.  One can refine a  partition by bisecting all boxes in `𝒫` along the $i$-th coordinate direction through 
```@repl 1
𝒫 = subdivide!(𝒫, i)
```

<!--The `BoxTree` created above is equivalent to a 4x2 `BoxGrid`. One can retrieve this using 
```@repl 1
P3 = BoxGrid(P2)
```
`BoxTree`s use indices of the type `(depth, cartesian_index)` where `cartesian_index` is the equivalent index of a `BoxGrid` with the same size as a `BoxTree` subdivided `depth` times. In other words,
```@repl 1
key_to_box( P, (1, 1) ) == key_to_box( P2, (4, (1, 1)) )
key_to_box( P, (4, 2) ) == key_to_box( P2, (4, (4, 2)) )
```-->

## BoxSet

A core idea of GAIO.jl is to approximate a subset of the domain $X$ via a collection of boxes. To construct a `BoxSet`, there are two typical options: retrieving all boxes from some partition, or locating a box surrounding a (or some) point(s) in $X$:
```@repl 1
ℬ = cover(𝒫, :)    # set of all boxes from 𝒫

x = (0.2, 0.4)
ℬ = cover(𝒫, x)    # set of one box containing the point x

xs = [ rand(2) for _=1:10 ]
ℬ = cover(𝒫, xs)   # set of boxes containing the points in xs
```

<!--One can also create a `Boxset` from an iterable of `Box`es. This will cover every element of the iterable with boxes from `P`:
```@repl 1
x1 = (0.2, 0.1)
box1 = point_to_box(P, x1)
x2 = (0.3, 0.6)
box2 = point_to_box(P, x2)
B = cover(P, [box1, box2])
```-->

You can access the boxes or their internal data via iteration
```@repl 1
for box in ℬ
    center, radius = box
    # do something
end

# get an array of boxes
arr_of_boxes = collect(ℬ)

# get an array of box centers
arr_of_centers = collect(box.center for box in ℬ)

# get an array of box radii
arr_of_radii = collect(box.radius for box in ℬ)
```

## BoxMap

A box map is a function which maps a `BoxSet` to a `BoxSet`. Given a map $f : \mathbb{R}^d\to \mathbb{R}^d$
```@repl 1
f((x,y)) = (1-1.4*x^2+y, 0.3*x)   # the Hénon map
```
and some box `X` as domain, you construct a `BoxMap` by
```@repl 1
F = BoxMap(f, X)
```
By default, GAIO.jl will try to adaptively choose sample points in a set to compute the image of the set by approximating the (local) Lipschitz constant of the map $f$. There are many other types of `BoxMap` discretizations available, see the [section on BoxMaps](boxmaps/boxmaps_general.md). 

We can now map a `BoxSet ℬ` via the `BoxMap F` by
```@repl 1
𝒞 = F(ℬ)
```
where the output `𝒞` is also a `BoxSet`.

For long running computations, you can also display a progress meter
```@repl 1
using ProgressMeter
𝒞 = F(ℬ; show_progress = true)
```

## TransferOperator

The _Perron-Frobenius operator_ (or _transfer operator_) [[lasotamackey](@cite)] associated to a map $f$ can be approximated using the `TransferOperator` type.  To construct a (discretized) `TransferOperator` from a `BoxMap` $F$ on the domain `BoxSet` `ℬ`, you type
```@repl 1
T = TransferOperator(F, ℬ)
```
Again, a progress meter can be displayed with the additional keyword argument `show_progress = true`.

<!--Internally, `GAIO.jl` will choose some enumeration of the domain and codomain, and use this for indexing the columns and rows, respectively.  You can generate the associated vector of boxes by `enumerated_ℬ = collect(T.domain)`. To convert the transfer operator to a matrix (cf. [[algGAIO](@cite)]), one can simply call the `sparse` function from `SparseArrays` 
```@repl 1
using SparseArrays

# mat[i, j] ≈ probability( f(x) ∈ enumerated_B[j]  |  x ∈ enumerated_B[i] )
mat = sparse(T)
```-->
Formally, `T` is a linear (in fact, a Markov) operator on the space of steps functions on the box set ℬ.  Of particular interest are often certain eigenvectors. For example, an eigenvector at the eigenvalue 1 is an (approximate) invariant measure, which characterizes the long term behaviour of $f$ according to [Birkhoff's ergodic theorem](https://en.wikipedia.org/wiki/Birkhoff%27s_ergodic_theorem). 
```@repl 1
λ, ev = eigs(T)
μ = ev[1]   # ev is an array of measures, grab the first one
```
This can also be done with the adjoint _Koopman operator_ `T'`. 

## BoxMeasure

The second output of `eigs(T)` is a vector of (discrete) measures, `BoxMeasure`s. A `BoxMeasure`  is absolutely continuous w.r.t. the volume (i.e. Lebesgue) measure and its density is piecewise constant on the boxes of the domain ℬ. One can let `T` act on a `BoxMeasure` simply through multiplication
```@repl 1
ν = T*μ
```
Of course, the same holds for the the Koopman operator as well. 
```@repl 1
ν = T'*μ
```
One can evaulate a `BoxMeasure` on an arbitrary `BoxSet` ℬ
```@repl 1
μ(ℬ)
```
Similarly, one can integrate a function with respect to a BoxMeasure via
```@repl 1
sum(x -> sin(x[1] + 2x[2]), μ)
```
Marginal distributions can be accessed using the `marginal` function
```@repl 1
marginal(μ; dim=1)
```
The measures can also be associated with a (Lebesgue) density
```@repl 1
p = density(μ)
```
Since a measure μ is a function defined over measurable sets, composite measures $g \circ \mu$ are well-defined for functions $g : \mathbb{R} \to \mathbb{R}$ (or $g : \mathbb{C} \to \mathbb{C}$). This is supported in GAIO.jl for `BoxMeasures`
```@repl 1
η = exp ∘ μ
```
Finite signed measures can be given a vector space structure. This is also supported:
```@repl 1
ν + μ
2ν - μ/2
```
<!--A `BoxMeasure` is implemented by a dictionary, mapping boxes to weights
```@repl 1
for (box, val) in μ
    println(box, " => ", val)
end
```
To access this structure oneself one can call
```
P = μ.partition
key_val_pairs = pairs(μ)
```-->

## Graphs of Boxes

One can equivalently view the transfer operator as a weighted directed graph. That is, the matrix of a `TrensferOperator` is the (transposed) weighted adjacency matrix for a graph. This graph can be constructed explicitely using the `MetaGraphsNext.jl` package 
```@repl 1
using Graphs, MetaGraphsNext
G = MetaGraph(T)
```
See also the [Graphs](https://juliagraphs.org/Graphs.jl/stable/) and [MetaGraphsNext](https://juliagraphs.org/MetaGraphsNext.jl/stable/) documentation. 

## Plotting

GAIO.jl offers both `Plots` or `Makie` for plotting. To plot a `BoxSet` or a `BoxMeasure`, simply choose either Plots or a Makie backend, eg. `GLMakie`, and call `plot` on a `BoxSet` or `BoxMeasure`
```julia
using GLMakie: plot

plot(ℬ)
plot(μ)
```
Plotting works with all the functionality of either package. This means you can set box plots as subplots, add colorbars, etc., using the Plots or Makie interface. For an example, see `examples/invariant_measure_2d.jl`. 
