# Conley Index

(TODO)

```@docs; canonical=false
index_pair
index_quad
@save
```

### Example

```@example 1
using GAIO 

# Henon map
const a, b = 1.4, 0.2
f((x,y)) = (1 - a*x^2 + y/5, 5*b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(:interval, f, P)
S = cover(P, :)
```

```@example 1
using SparseArrays, LinearAlgebra
function period_n_orbit(F, B; n=2)
    F♯ = TransferOperator(F, B, B)

    M = sparse(F♯)         # transfer matrix
    N = F♯.domain

    for _ in 2:n
        M .= F♯.mat * M    # Mⁿ :  n-fold transfer matrix
    end
                           # nonzero diagonal elements are
    v = diag(M)            # (candidates for) fixed points under n-fold iteration
    BoxSet(N, v .> 0)
end
```

```@example 1
A = maximal_invariant_set(F, S, steps = 20)
per = [period_n_orbit(F, A; n=n) for n in 1:6]
```

```@example 1
B = union(per[2:end]...)
B = setdiff!(B, per[1])     # periodic points, excluding fixed points
```

```@example 1
N = isolating_neighborhood(F, B)
```

```@example 1
P1, P0 = index_pair(F, N)
```

Compute pairs to construct the index map ``F:\ (P_1,\ P_0) \to (Q_1,\ Q_0)``

```@example 1
P1, P0, Q1, Q0 = index_quad(F, N)
```

```@example 1
transfers = TransferOperator(F, P1, Q1)
```

```@example 1
@save P1 
```

```@example 1
@save P0
@save Q1
@save Q0
@save transfers
```

```@example 1
using Plots

p = plot(P1)
p = plot!(p, P0, color=:blue)

savefig(p, "P.svg"); nothing # hide
```

![domain pair](P.svg)

```@example 1
p = plot(Q1)
p = plot!(p, Q0, color=:blue)

savefig(p, "Q.svg"); nothing # hide
```

![image pair](Q.svg)

Get `homcubes` at [Pawel Pilarczyk's website](http://www.pawelpilarczyk.com/chomp/software/)


```
$ homcubes -g P1_generators.dat -g Q1_generators.dat -g graph_generators.dat transfers.map P1.cub P0.cub Q1.cub Q0.cub

HOMCUBES, ver. 3.07, 09/25/15. Copyright (C) 1997-2020 by Pawel Pilarczyk.
This is free software. No warranty. Consult 'license.txt' for details.
Note: The empty file 'P0.cub' is assumed to contain cubes.
Reading cubes to X from 'P1.cub'... 334 cubes read.
Reading cubes to A from 'P0.cub'... 0 cubes read.
Reading cubes to Y from 'Q1.cub'... 574 cubes read.
Reading cubes to B from 'Q0.cub'... 240 cubes read.
Computing Y\B... 240 cubes removed from Y, 334 left.
300 bit fields allocated (0 MB) to speed up 2-dimensional reduction.
Reducing full-dim cubes from X... 314 removed, 20 left.
Note: The program assumes that the input map is acyclic.
Reading the map on X from 'transfers.map'... Done.
Verifying if the image of X is contained in Y... Passed.
Computing the image of the map... 63 cubes.
Expanding B in Y... 273 cubes moved to B, 61 left in Y\B.
Restricting B to the neighbors of Y\B... 401 cubes removed, 112 left.
Reducing full-dim cubes from (Y,B)... 67 removed, 106 left.
Transforming X into cells... 20 cells added.
Transforming Y\B into cells... 22 cells added.
Transforming B into cells... 84 cells added.
Collapsing faces in X... .. 160 removed, 20 left.
Note: The dimension of X decreased from 2 to 0.
Creating the map F on cells in X... 63 cubes added.
Creating a cell map for F... . Done.
Note: It has been verified successfully that the map is acyclic.
Creating the graph of F...  20 cells added.
Adding boundaries of cubical cells in Y and B... 91 cubical cells added.
Forgetting 256 cells from B.
Computing the image of F... 2 cells added.
Collapsing Y towards F(X)... .. 86 cells removed, 27 left.
Note: The dimension of Y decreased from 2 to 1.
Creating the chain complex of the graph of F... Done.
Creating the chain complex of Y... . Done.
Creating the chain map of the projection... Done.
Time used so far: 0.00 sec (0.000 min).
Computing the homology of the graph of F over the ring of integers...
H_0 = Z^20
Saving generators of X to 'P1_generators.dat'... Done.
Saving generators of the graph of F to 'graph_generators.dat'... Done.
Computing the homology of Y over the ring of integers...
Reducing D_1: 0 + 4 reductions made. 
H_0 = Z^2
H_1 = Z^17
Saving generators of Y to 'Q1_generators.dat'... Done.
The map induced in homology is as follows:
Dim 0:  f (x1) = 0
        f (x2) = 0
        f (x3) = 0
        f (x4) = y1
        f (x5) = 0
        f (x6) = 0
        f (x7) = y2
        f (x8) = 0
        f (x9) = 0
        f (x10) = 0
        f (x11) = 0
        f (x12) = 0
        f (x13) = 0
        f (x14) = 0
        f (x15) = 0
        f (x16) = 0
        f (x17) = 0
        f (x18) = 0
        f (x19) = 0
        f (x20) = 0
Total time used: 0.00 sec (0.000 min).
Thank you for using this software. We appreciate your business.
```

We can take a look at the generators in homology within the *_generators.dat files:

```
$ cat Q1_generators.dat

The 2 generators of H_0 follow:
generator 1
1 * [(703,542)(703,542)]
generator 2
1 * [(700,543)(700,543)]

The 17 generators of H_1 follow:
generator 1
1 * [(455,683)(456,683)]
generator 2
1 * [(621,403)(622,403)]
generator 3
1 * [(394,702)(395,702)]
generator 4
1 * [(629,408)(630,408)]
generator 5
1 * [(546,632)(547,632)]
generator 6
-1 * [(614,618)(615,618)]
1 * [(615,617)(615,618)]
generator 7
1 * [(314,313)(315,313)]
generator 8
1 * [(401,695)(402,695)]
generator 9
1 * [(568,619)(569,619)]
generator 10
-1 * [(653,590)(654,590)]
1 * [(653,590)(653,591)]
generator 11
1 * [(546,649)(547,649)]
generator 12
1 * [(447,677)(448,677)]
generator 13
1 * [(590,605)(591,605)]
generator 14
1 * [(604,393)(605,393)]
generator 15
1 * [(411,697)(412,697)]
generator 16
1 * [(692,458)(692,459)]
1 * [(692,459)(693,459)]
generator 17
1 * [(676,442)(676,443)]
1 * [(675,442)(676,442)]
```
