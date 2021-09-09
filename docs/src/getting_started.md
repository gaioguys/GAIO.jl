# Getting started

Consider the famous [Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) [1]
```math
f(x,y) = (1-ax^2+y, bx), \quad a,b \in \mathbb{R}
```
Iterating some random intial point exhibits a, well, ''strange'' attractor (a=1.4 and b=0.3)
![GitHub Logo](../henon-simulation.svg)

Since this map is _chaotic_ [2,3], it has sensitive dependence on initial conditions.  That is, small perturbations (as unavoidable on a computer) during the computation grow exponentially during the iteration.  Thus, apart from a few iterates at the beginning, the computed trajectory does not (necessarily) follow a true trajectory. One might therefore question how reliable this figure is.

Instead of trying to approximate the attractor by a long forward trajectory, we will capture it by computing a collection of boxes covering it.

Start by loading the GAIO package
```julia
using GAIO
```
A `Box` is descibed by its center and its radius
```julia
center, radius = (0,0), (3,3)
Ω = Box(center, radius)
```
This box serves as the domain for our computation.

From some box Ω, a `BoxSet` can be constructed. A `BoxSet` is a subset of a partition of Ω into smaller boxes:
```julia
B = BoxSet(Ω, (4,4)) 
```
yields an empty `BoxSet` as a subset from the partition of Ω into a grid of 4 x 4 equally sized smaller boxes.

In order to deal with the Hénon map as a map on box sets, we have to turn it into a `BoxMap`
```julia
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x) 
F = BoxMap(f, B) 
```
We can now compute a covering of the attractor
```julia
A = relative_attractor(F, B[:], steps = 15)  
plot(A)
```
using the subdivison algorithm described in [4]. The command `B[:]` returns the box set containing all boxes from the underyling 4 x 4 partition.  Note that we could as well have started the computation with `B = BoxSet(Ω)` which is a shortcut for `B = BoxSet(Ω, (1,1))`, i.e. the ''trivial'' partition consisting of a single box (i.e. Ω) only. 

In addition to covering the attractor, this box collection also covers an unstable fixed point near (-1,-0.3) and its unstabe manifold (cf. [4]).

![GitHub Logo](../henon-attractor.svg)



## References

[1] Hénon, Michel. "A two-dimensional mapping with a strange attractor". Communications in Mathematical Physics 50.1 (1976): 69–77.

[2] Benedicks, Michael, and Lennart Carleson. "The dynamics of the Hénon map." Annals of Mathematics 133.1 (1991): 73-169.

[3] Zgliczynski, Piotr. "Computer assisted proof of chaos in the Rössler equations and in the Hénon map." Nonlinearity 10.1 (1997): 243. 

[4] Dellnitz, Michael, and Andreas Hohmann. "A subdivision algorithm for the computation of unstable manifolds and global attractors." Numerische Mathematik 75.3 (1997): 293-317.