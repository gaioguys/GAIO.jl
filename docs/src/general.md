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

## BoxGrid

Most algorithms in GAIO.jl revolve around a partition of the domain ``Q`` into small boxes. To create an ``n_1 \times \ldots \times n_d`` - element equidistant grid of boxes, we can pass the tuple ``n = (n_1, \ldots, n_d)`` into the function `BoxGrid`
```@repl 1
n = (4, 2)
P = BoxGrid(Q, n)
```

`BoxGrid`s use a cartesian indexing structure to be memory-efficient. These indices are accessed and used through the API:
```@repl 1
x = (0.2, 0.1)
key = point_to_key(P, x)    # x is some point in the domain Q
box = key_to_box(P, key)    # cover the point x with a box from P
box = point_to_box(P, x)    # performs both above functions
```

## BoxTree

For partitions of ``Q`` into variably sized boxes, one can use `BoxTree`:
```@repl 1
P2 = BoxTree(Q)
```
A `BoxTree` uses a binary tree structure to store a partition of the domain. Every Box of a `BoxTree` can be split using the command 
```@repl 1
subdivide!(P2)
subdivide!(P2)
subdivide!(P2)
```
The axis direction along which to subdivide cycles through with the depth, i.e. subdividing at depth 1 splits along dimension 1, subdividing at depth `d+1` splits along dimension 1 again. 

The `BoxTree` created above is equivalent to a 4x2 `BoxGrid`. One can retrieve this using 
```@repl 1
P3 = BoxGrid(P2)
```
`BoxTree`s use indices of the type `(depth, cartesian_index)` where `cartesian_index` is the equivalent index of a `BoxGrid` with the same size as a `BoxTree` subdivided `depth` times. In other words,
```@repl 1
key_to_box( P, (1, 1) ) == key_to_box( P2, (4, (1, 1)) )
key_to_box( P, (4, 2) ) == key_to_box( P2, (4, (4, 2)) )
```

## BoxSet

The core idea behind GAIO.jl is to approximate a subset of the domain via a collection of small boxes. To construct `BoxSet`s, there are two typical options: getting all boxes in the partition, or locating a box surrounding a point ``x \in Q``
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

`BoxSet` is a meant to be a memory-efficient way of storing boxes. However, should you want to access the boxes or their internal data, this can be done via iteration:
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

A BoxMap is a function which maps boxes to boxes. Given a pointmap `f : ℝᵈ → ℝᵈ` which accepts an `SVector` from `StaticArrays.jl` (or just an `NTuple`) and returns the same, 
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
This will generate a `BoxMap` which attempts to calculate setwise images of `f`. One can also use a `DynamicalSystem` from the very good `DynamicalSystems.jl`: 
```@repl 1
using DynamicalSystems: DiscreteDynamicalSystem
using StaticArrays

dynamic_rule(u::Vec, p, t) where Vec = Vec( f(u) ) # DynamicalSystems.jl syntax

u0 = SA_F64[0, 0]   # this is dummy data
p0 = SA_F64[0, 0]   # parameters, in case they are needed

system = DiscreteDynamicalSystem(dynamic_rule, u0, p0)

F̃ = BoxMap(system, Q)
```
The same works for a continuous dynamical system. 

!!! warning "Maps based on `DynamicalSystem`s cannot run on the GPU!"
    Currently, you must hard-code your systems, and cannot rely on `DifferentialEquations` or `DynamicalSystems` for GPU-acceleration. 

!!! warning "Check your time-steps!"
    GAIO.jl ALWAYS performs integration over **one** time unit for `ContinuousDynamicalSystem`s! To perform smaller steps, rescale your dynamical system accordingly!

```@repl 1
using DynamicalSystems: ContinuousDynamicalSystem
using OrdinaryDiffEq

function lorenz_dudx(u::Vec, p, t, Δt) where Vec
    x,y,z = u
    σ,ρ,β = p

    dudx = Vec(( σ*(y-x), ρ*x-y-x*z, x*y-β*z ))
    return Δt * dudx
end

Δt = 0.2
lorenz(u, p=p0, t=0) = lorenz_dudx(u, p, t, Δt)

u0 = SA_F64[0, 0, 0]
p0 = SA_F64[10, 28, 0.4]
diffeq = (alg = RK4(), dt = 1/20, adaptive = false)

lorenz_system = ContinuousDynamicalSystem(lorenz, u0, p0; diffeq)

Q̄ = Box((0,0,25), (30,30,30))
F̄ = BoxMap(lorenz_system, Q̄)
```
By default, GAIO.jl will try to adaptively choose test points to compute setwise images by approximating the (local) Lipschitz constant of the map. There are many other types of `BoxMap` discretizations available, see the section on BoxMaps for more information. 

## Using BoxMap

Now, one can map a `BoxSet` via the `BoxMap` `F` by simply calling `F` as a function 
```@repl 1
C = F(B)
```
where the output `C` is also a `BoxSet`.

For long running computations, GAIO.jl can also display a progress meter
```@repl 1
using ProgressMeter
C = F(B; show_progress = true)
```
(Adding a progress meter adds a little bit of overhead, so for super short computations like the above it isn't recommended)

## TransferOperator

The _Perron-Frobenius operator_ (or _transfer operator_) [lasotamackey](@cite) is discretized in GAIO.jl using the `TransferOperator` type. To initialize a `TransferOperator` that acts on a domain `BoxSet` `B` and codomain `BoxSet` `C` 
```@repl 1
T = TransferOperator(F, B, C)   # T operates on the domain covered by the box set B
```
Most often (e.g. in eigenvalue calculations), one wants the domain and codomain to be equal 
```@repl 1
B = cover(P, :)
T = TransferOperator(F, B, B)
```
Again, a progress meter can be displayed for long computations
```@repl 1
using ProgressMeter
T = TransferOperator(F, B, B; show_progress = true)
```
Internally, `GAIO.jl` will choose some enumeration of the domain and codomain, and use this for indexing the columns and rows, respectively. 
```@repl 1
enumerated_B = collect(T.domain)
```
To convert the transfer operator to the underlying transfer matrix described in [algGAIO](@cite), one can simply call the `sparse` function from `SparseArrays` 
```@repl 1
using SparseArrays

# mat[i, j] ≈ probability( f(x) ∈ enumerated_B[j]  |  x ∈ enumerated_B[i] )
mat = sparse(T)
```
To find an approximate invariant measure over `B` use the `eigs` function from `Arpack.jl`. All keyword arguments from `Arpack.eigs` are supported. 
```@repl 1
# for the Baker trafo, the Lebesgue measure 
# - i.e. the constant-weight measure - is invariant
λ, ev = eigs(T);
λ
ev
μ = ev[1]   # ev is an array of measures, grab the first one
```
This can also be done with the adjoint _Koopman operator_ `T'`. 

## BoxMeasure

The second output of `eigs(T)` is a discrete measure, a `BoxMeasure`. This measure is absolutely continuous w.r.t. the volume (i.e. Lebesgue) measure and its density is piecewise constant on the boxes of the domain. One can let `T` act on a `BoxMeasure` simply through multiplication
```@repl 1
ν = T*μ
```
Of course, the same holds for the the Koopman operator as well. 
```@repl 1
ν = T'μ
```
One can evaulate a `BoxMeasure` on a `BoxSet`
```@repl 1
B
μ(B)
```
Similarly, one can integrate a function with respect to a BoxMeasure by calling
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
Since a measure ``\mu`` is a function defined over measurable sets, composite measures ``g \circ \mu`` are well-defined for functions ``g : \mathbb{R} \to \mathbb{R}`` (or ``g : \mathbb{C} \to \mathbb{C}``). This is supported in GAIO.jl for `BoxMeasures`
```@repl 1
η = exp ∘ μ
```
For multiple BoxMeasures, the concatenation operator `∘` can be applied to each one using julia's broadcasting functionality
```@repl 1
real_ev = real .∘ ev
```
Similarly, finite signed measures can be given a vector space structure. This is also supported in GAIO.jl
```@repl 1
ν + μ
2ν - μ/2
```
A `BoxMeasure` is implemented by a dictionary, mapping boxes to weights
```@repl 1
for (box, val) in μ
    println(box, " => ", val)
end
```
To access this structure oneself one can call
```
P = μ.partition
key_val_pairs = pairs(μ)
```

## Graphs of Boxes

One could equivalently view the transfer operator as a weighted directed graph. That is, a transfer matrix in GAIO.jl is the (transposed) weighted adjacency matrix for a graph. This graph can be constructed using the `MetaGraphsNext.jl` package 
```@repl 1
using Graphs, MetaGraphsNext
G = MetaGraph(T)
```
See also the [Graphs](https://juliagraphs.org/Graphs.jl/stable/) and [MetaGraphsNext](https://juliagraphs.org/MetaGraphsNext.jl/stable/) documentation. 

## Plotting

GAIO.jl offers both `Plots` or `Makie` for plotting. To plot a `BoxSet` or a `BoxMeasure`, simply choose either Plots or a Makie backend, eg. `GLMakie`, and call `plot` on a `BoxSet` or `BoxMeasure`
```julia
using GLMakie: plot

plot(B)
```
Plotting works with all the functionality of either package. This means you can set box plots as subplots, add colorbars, etc., using the Plots or Makie interface. For an example, see `examples/invariant_measure_2d.jl`. 
