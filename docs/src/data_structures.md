# Data structures

## Box

A single box can be created by specifying its center and its radius as vectors (or tuples) in `\mathbb{R}^d`:
```julia
center = [1, 2.5]
radius = [0.5, 1.0]
Q = Box(center, radius)
plot(Q)
```
<p><img src="images/box.svg" alt="Hénon attractor" width=60%/></p>

You can compute the volume of a box
```julia
julia > volume(Q)
2.0
```
and check whether it contains some point
```julia
julia> x = rand(2)
julia> x ∈ Q
false
```

## BoxPartition

A `BoxPartition` specifies a partition of some box into smaller boxes. A box partition can be constructed from a given (''outer'') box Q. The command  
```julia
P = BoxPartition(Q, (10,20))
```
returns a partition of Q into 10 x 20 equally sized boxes. 

## BoxSet
From a box partition, sets of boxes can be constructed. The command
```julia
B = P[:]
```
returns a `BoxSet` B containing all boxes from the underlying 10 x 20 partition. On the other end, 
```julia
E = empty(P)
```
yields the empty box set (as a subset of the partition P). You can test whether some box set S is empty by `isempty(S)`.

You can also construct a box set containing a given set X of points by entering
```julia
X = [ 3*rand(2) for _ = 1:100 ]
C = P[X]
```
Box sets are Julia sets (in the sense of a data struture of the Julia language ...) and so all common commands for sets and collections can be applied. For example, you can get a vector which explicitely lists all boxes from B by
```julia
v = collect(B)
```
This can be used in order to get access to the geometry of the boxes in B:
```julia
julia> b = v[1]
julia> b.center, b.radius
([1.45, 2.55], [0.05, 0.05])
```
You can take the union, intersection and the difference of box sets
```julia
julia> B ∪ C
200-element BoxSet in dimension 2
julia> B ∩ C
45-element BoxSet in dimension 2
julia> setdiff(B, C)
155-element BoxSet in dimension 2
```
and query the number of elements
```julia
julia> length(B)
200
```
Finally, you can _subdivide_ a box set with respect to some dimension k ∈ 1:d.  The command
```julia
Ĉ = subdivide(C,k)
```
returns a box set Ĉ which results from bisecting each box in C with respect to the k-th coordinate direction. Note that Ĉ is an element of a new box partition. This can be queried by
```julia
julia> Ĉ.partition
20 x 20 BoxPartition
```


