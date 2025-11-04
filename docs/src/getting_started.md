# Getting started

Consider the [Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) [henon](@cite)

```@repl 1
const a, b = 1.35, 0.3
f((x,y)) = ( 1 - a*x^2 + y,  b*x ) 
```

Iterating some random intial point exhibits a strange attractor 

```@repl 1
x = (1., 1.)
orbit = [x]
for k in 1:10000
    x = f(x)
    push!(orbit, x)
end

using Plots
p = scatter(orbit)
p = scatter(orbit, dpi=500) # hide
savefig(p, "henon-simulation.png"); nothing # hide
```

![Hénon attractor](henon-simulation.png)

This map is _chaotic_ [henonchaos1](@cite), [henonchaos2](@cite), it has sensitive dependence on initial conditions. That is, small perturbations (as unavoidable on a computer) during the computation grow exponentially during the iteration.  Thus, apart from a few iterates at the beginning, the computed trajectory does not (necessarily) follow a true trajectory. One might therefore question how reliable this figure is.

Instead of trying to approximate the attractor by a long forward trajectory, we will capture it by computing a collection of boxes (i.e. cubes) covering the attractor. 

Start by loading the GAIO package
```@repl 1
using GAIO
```
A `Box` is descibed by its center and its radius
```@repl 1
box_center, box_radius = (0,0), (3,3)
X = Box(box_center, box_radius)
```
This box will serve as the domain for our computation.  The box covering which we will compute is a subset of a _partition_ of `X` into smaller boxes. The command
```@repl 1
P = BoxGrid(X, (4,4)) 
```
yields a partition of `X` into a grid of 4 x 4 equally sized smaller boxes. Note that this command does not explicitly construct the partition (as a set of subsets covering the domain `X`) but rather serves as a ``\sigma``-algebra for constructing sets of boxes later. For example, the commands
```@repl 1
points = [
    (1, 1),
    (2, 1)
];
B = cover(P, points)
```
yields a `BoxSet` containing boxes from the partition `P` which cover each of `points`. Similarly, 
```@repl 1
B = cover(P, :)
```
yields a `BoxSet` containing all boxes from the partition `P` (i.e. a set containing 16 boxes).

In order to deal with the Hénon map `f` as a map over box sets, we have to turn it into a `BoxMap` on the domain `X`
```@repl 1
F = BoxMap(f, X) 
```
We can now compute a covering of the attractor in `X`, starting with the full box set `B`, by applying a couple of steps of the subdivison algorithm described in [subalg](@cite):
```@repl 1
A = relative_attractor(F, B, steps = 19)  
```

```@repl 1
p = plot(A)
p = plot(A; dpi=500) # hide
savefig(p, "henon.png"); nothing # hide
```

![box covering of the Hénon attractor](henon.png)

In addition to covering the attractor, this box collection also covers an unstable fixed point near (-1,-0.3) and its unstabe manifold (cf. [subalg](@cite)). 
