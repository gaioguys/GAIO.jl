# General usage
The base of the numerical set oriented methods of this framework are `BoxSet` (the discretization of a set) and `BoxMap` (the discretization of a map), thus, in the following, we will have a closer look at the two and other useful things to know when working with GAIO.

To create a `Box` given its center point `C` as well as its extent in every axis direction `(t_1, t_2, ...)`, simply do
```julia
box = Box(C, (t_1, t_2, ...))
```

## BoxSet
The idea is to approximate an `AbstractSet`via a collection of small boxes.
Thus, given an `AbstractSet` `S` construct the 'discretized set' `boxset` as 
```julia
boxset = BoxSet(P,S)
```
where `P` denotes the underlying `BoxPartition`, that is the information about how small the boxes we use to approximate `S` with are. The `BoxPartition` is always constructed for a larger `Box` completely containing `S`.

In general one has two choices to construct the partition:
* Using a **regular partition**, that is a partition where the small boxes all have the same size. 
  Given a `Box` `box`, do
  ```julia
  partition = RegularPartition(box, d)
  ```
  where `d` is an `Int` corresponding to the size of the boxes: The partition will contain ``2^d`` boxes, i.e. every box in the partition is ``\frac{1}{2^d}`` times the size of the original box `box`.
* Using a **tree partition**
  > [TODO]
 
## BoxMap
Given a pointmap `f`, initialize the corresponding `BoxMap` `g` by
```julia
g = PointDiscretizedMap(f, points)
```
where `points` is the set of reference points, implemented as an `Array` of `Tuples`s, describing the discretization of the ``n``-dimensional reference box ``[-1,1]^n``

Now, one can map a `BoxSet` `boxset` via the `BoxMap` `g` by
```julia
boximage = g(boxset)
```
where the output `boximage`is also a `BoxSet`.
Optionally, one can choose a different image domain, e.g. if `boximage` needs to have a different `Partition` than `boxset`. In this case, given `boxset_new`, the `BoxSet`with the `Partition`, we want `boximage` to have, do
```julia
boximage = g(boxset; target = boxset_new)
```

## The Subdivision Algorithm

## Miscellaneous
To plot a `BoxSet` `boxset` simply do
```julia
plot(boxset)
```
> [TODO: Options for plot]

