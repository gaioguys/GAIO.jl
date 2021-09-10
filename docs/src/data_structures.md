# Data structures

## Box

A single box can be created by specifying its center and its radius as vectors (or tuples) in `\mathbb{R}^d`:
```julia
center = [1, 2, 3.5]
radius = [0.5, 1.0, 2.0]
box = Box(center, radius)
plot(box)
```
<p><img src="images/box.svg" alt="Hénon attractor" width=60%/></p>

You can compute the volume of a box
```julia
julia > volume(box)
2.0
```
and check whether it contains some point
```julia
julia> x = rand(2)
julia> x ∈ box
false
```

## BoxSet

A `BoxSet` is a subset of a partition of some box into smaller boxes. A box set can be constructed from a given (''outer'') box `Ω`. The command  
```julia
B = BoxSet(Ω, (10,20))
```
returns the empty box set as a subset of the partition of `Ω` into 10 x 20 equally sized boxes. 
