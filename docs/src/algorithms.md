# Algorithms and Mathematical Background
!!! note "Note"
    In the following, ``f`` will always refer to the map describing the dynamics of a system, while `g` will be the corresponding `BoxMap`.


## The Relative Global Attractor

Consider a time discrete dynamical system induced by the map ``f: \mathbb{R}^d \to \mathbb{R}^d``. Let ``Q \subset \mathbb{R}^d`` compact.
The set 
```math
A_Q := \bigcap_{k \geq 0} f^k(Q)
```
is called the global attractor relative to ``Q``.
The relative global attractor can be seen as the set which is eventually approached by every orbit originating in ``Q``. In particular, ``A_Q`` contains each invariant set in ``Q`` and therefore all the potentially interesting dynamics. Thus it is of great interest to be able to compute a relative global attractor numerically. 
The idea of the algorithm is to cover the relative global attractor with boxes and recursively tighten the covering by refining appropriately selected boxes.

### Mathematical Background of the Algorithm
Mathematically, the algorithm to compute the global attractor relative to ``Q`` takes two input arguments: a compact set ``Q`` as well as a map ``f``, which describes the dynamics. Now in each iteration, two steps happen:
1. **subdivision-step:** The domain ``B_{k-1}`` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, ``\hat{B}_k``, with double the amount of boxes.
2. **selection_step:** For each box ``B`` of the new partition we check, if there is another Box ``B'`` that is mapped into ``B`` under `g`, i.e. if ``f(B') \cap B \neq \emptyset``. If not, we remove ``B`` from the domain.

After removing every non-hit box, we arrive at the new domain ``B_k``, and as ``k \to \infty``, the collection of boxes ``B_k`` converges to the relative global attractor ``A_Q``.

### Implementation of the Algorithm
So, how was taken care of the discretization, which is necessary for the implementation? In other words, how did we translate this Algorithm into GAIO?
```julia
function relative_attractor(boxset::BoxSet, g::BoxMap, depth::Int)
    for k = 1:depth
        boxset = subdivide(boxset)
        boxset = g(boxset; target=boxset)
    end

    return boxset
end
```
The first thing one notices is that the implementation has a third input parameter `depth`, which describes the level of approximation. Since in each step of the algorithm the initial domain is divided in half, the final partition after `depth` many steps will contain ``n := 2^{\text{depth}}`` boxes, i.e. every box in the final covering is ``\frac{1}{n}`` times the size of the initial box. Besides `depth` we have the input parameters `boxset` and `g`, which is the `BoxMap` describing the dynamics, that is a function which maps boxes to boxes.
`g` is going to be implemented as a `PointDiscretizedMap`, which is a `struct` containing the underlying pointwise map corresponding to the dynamics as well as the set of reference points, implemented as an `Array` of `Tuple`s, that will be the discretization of a box.
```julia
struct PointDiscretizedMap{F,P} <: BoxMap
    f::F
    points::P
end
```
`boxset` is `struct`, which carries both a set and the partition of the set, that is the information about the size of each single box in the set.
```julia
struct BoxSet{P <: BoxPartition,S <: AbstractSet}
    partition::P
    set::S
end
```
For this algorithm the partition needs to be the full regular partition of the initial set (i.e. the starting domain).
Now, for each step in the algorithm, the current set is subdivided via the subdivision algorithm:
```julia
boxset = subdivide(boxset)
```
And it is checked, if a box is hit by another box under the dynamics:
```julia
function map_boxes_with_target(g, source::BoxSet, target::BoxSet)
    result = boxset_empty(target.partition)

    for (_, hit) in ParallelBoxIterator(g, source, target.partition)
        if hit !== nothing # check that point was inside domain
            if hit in target.set
                push!(result.set, hit)
            end
        end
    end

    return result
end
``` 
where implementationally we start with an empty boxset and store each box that is hit in it.

---

## unstable_set
In the following we are presenting the algorithm to cover invariant manifolds within some domain ``Q`` (which has to contain a fixed point).
!!! note "Note"
    For simplicity, we will explain the algorithm for the case of the *unstable manifold*. However one can compute the stable manifold as well by considering the boxmap describing the inverse map ``f^{-1}`` as input argument for the algorithm.
    
Usually, the computation of the unstable manifold is relatively simple in 1D, but the higher the dimension, the more complicated it becomes. GAIO is able to compute the unstable manifold for arbitrary dimension.
    
### Mathematical Background of the Algorithm
The unstable manifold is defined as
```math
W^U(x_0) = \{x: \lim_{k \to - \infty} f^k(x) = x_0 \}
```
where ``x_0`` is a fixed point of ``f``.

The idea behind the algorithm to compute the unstable manifold can be explained in two steps. Before starting we need to identify a hyperbolic fixed point and the region ``Q``, which we are going to compute the manifold in. The region ``Q`` needs to be already partitioned into small boxes.
1. Since a fixed point is always part of the unstable manifold, we need to identify a small region/box containing this fixed point.
2. The small box containing the fixed point is then mapped forward under the dynamics defined by ``f`` and the boxes that are hit under the image are added to the box collection. Then those newly included boxes are mapped forward and the procedure is repeated. 

With these two steps we obtain a covering of part of the global unstable manifold.

!!! warning "Warning"
    One might not be able to compute the parts of the unstable manifold whose preimage lies outside the domain ``Q``.
    Thus, it is important to choose ``Q`` large enough.

How was this algorithm translated into the language of GAIO?

### Implementation of the Algorithm
    

```julia
function unstable_set!(boxset::BoxSet, g::BoxMap)
    boxset_new = boxset

    while !isempty(boxset_new)
        boxset_new = g(boxset_new)

        setdiff!(boxset_new, boxset)
        union!(boxset, boxset_new)
    end

    return boxset
end
```
Let us start with the input arguments for the algorithm:
Again, like in the relative attractor Algorithm from above, `g` is the `BoxMap` describing the underlying dynamics ``f``. It thus stores ``f`` and a set of reference points necessary for the discretization of the boxes.
The other input argument ``boxset`` includes two things:
1. The domain ``Q`` we are going to compute the unstable manifold in (``Q`` can be implemented as a large `Box`) and the underlying partition of the domain. Unlike in the previous algorithm, the domain will not be subdivided along the algorithms course, but we need to pass a partition which is already subdivided to the depth ``d`` (and therefore the level of accuracy) we want our final boxcovering to have.
2. Since `boxset` is going to store all the new boxes we aquire in every iteration of the algorithm, it has to be initialized containing no other box than the single box of size ``\frac{1}{2^d}`` around the fixed point that is part of the unstable manifold we intend to compute.

Note: This algorithm works with two mutable sets of boxes: `boxset`, which collects the boxes we aquire in each iteration and will eventually cover part of the unstable manifold, and `boxset_new`, which will be overwritten in each iteration and contains only the boxes which will be newly added to our collection.
To initialize, we set
```julia
boxset_new = boxset
```
Now we repeat the following steps: 
First, we map the newly aquired boxes one step forward in time
```julia
boxset_new = g(boxset_new)
```
Note: Mapping only the newly acquired boxes from the previous step saves memory and computation time since we already computed the images of the old boxes in previous steps and thus those boximages are already part of the collector `boxset`.
Now we need to update `boxset_new` and `boxset`.
As mentioned prior, we only want to consider boxes in each iteration step, that have not been 'hit' under `g` by any boxes we acquired in a previous iteration step, because that would mean that this box image already is part of our box collection. To differ between truly new boxes and boxes we already added, we take the setdifference between the images of boxes `boxset_new` and the whole boxcollection `boxset`:
```julia
setdiff!(boxset_new, boxset)
```
Now `boxset_new` contains nothing but the truly new boxes.
`boxset` is then updated by adding those new image boxes to our collection of boxes by forming the union with the already existing collection:
```julia
union!(boxset, boxset_new)
```
We repeat these steps as long as
```julia
while !isempty(boxset_new)
```
is true. Thus, the iteration will end when no new boxes can be added to the boxcollection, because we e.g. got so close to the border of the domain ``Q``, that every further image of boxes lies beyond the border, or the unstable manifold oscillates so strongly, that our chosen level of accuracy can no longer distinguish between the oscillations.

---
## transition_matrix

---
## chain_recurrent_set

---
## root_covering
