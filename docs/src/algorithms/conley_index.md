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
A = relative_attractor(F, S, steps = 20)
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
Reading cubes to X from 'P1.cub'... 8662 cubes read.
Reading cubes to A from 'P0.cub'... 566 cubes read.
Computing X\A... 0 cubes removed from X, 8662 left.
Restricting A to the neighbors of X\A... 0 cubes removed, 566 left.
Reading cubes to Y from 'Q1.cub'... 8662 cubes read.
Reading cubes to B from 'Q0.cub'... 760 cubes read.
Computing Y\B... 0 cubes removed from Y, 8662 left.
300 bit fields allocated (0 MB) to speed up 2-dimensional reduction.
Reducing full-dim cubes from (X,A)... 179 removed, 9049 left.
Reading the map on X\A from 'transfers.map' for extended reduction... Done.
Verifying if the image of X\A is contained in Y... Passed.
Expanding A in X... 6590 moved to A, 2072 left in X\A, 7794 added to B.
Restricting A to the neighbors of X\A... 5287 cubes removed, 1690 left.
Reducing full-dim cubes from (X,A)... 490 removed, 3272 left.
Note: The program assumes that the input map is acyclic.
Reading the map on X\A from 'transfers.map'... Done.
Reading the map on A from 'transfers.map'... Done.
Verifying if the image of A is contained in B... Failed.
WARNING: The image of A is NOT contained in B.
Verifying if the image of A is disjoint from Y\B... Failed.
SERIOUS WARNING: The image of A is NOT disjoint from Y\B.
Computing the image of the map... 4590 cubes.
Expanding B in Y... 867 cubes moved to B, 1 left in Y\B.
Restricting B to the neighbors of Y\B... 4832 cubes removed, 4589 left.
Reducing full-dim cubes from (Y,B)... 0 removed, 4590 left.
Transforming X\A into cells... 2072 cells added.
Transforming A into cells... 1200 cells added.
Transforming Y\B into cells... 1 cells added.
Transforming B into cells... 4589 cells added.
Collapsing faces in X and A... .. 0 removed, 10104 left.
There are 7153 faces of dimension up to 2 left in A.
Creating the map F on cells in X... 105698 cubes added.
Creating the map F on cells in A... 56344 cubes added.
Creating a cell map for F... ... Done.     
Note: It has been verified successfully that the map is acyclic.
Creating the graph of F... .. 55227 cells added.
Adding boundaries of cubical cells in Y and B... 0 cubical cells added.
Forgetting 5375 cells from B.
Computing the image of F... 1 cells added.
Collapsing Y towards F(X)... .. 0 cells removed, 1 left.
Creating the chain complex of the graph of F... .. Done.
Creating the chain complex of Y... .. Done.
Creating the chain map of the projection... Done.
Time used so far: 0.87 sec (0.014 min).
Computing the homology of the graph of F over the ring of integers...
Reducing D_2: 0 + 15428 reductions made. 
Reducing D_1: 6977 + 1881 reductions made. 
H_0 = 0
H_1 = 0
H_2 = Z
Saving generators of X to 'P1_gen.dat'... Done.
Saving generators of the graph of F to 'graph_gen.dat'... Done.
Computing the homology of Y over the ring of integers...
Reducing D_2: 
Reducing D_1: 
H_0 = 0
H_1 = 0
H_2 = Z
Saving generators of Y to 'Q1_gen.dat'... Done.
The map induced in homology is as follows:
Dim 0:  0
Dim 1:  0
Dim 2:  f (x1) = y1
Total time used: 1.09 sec (0.018 min).
Thank you for using this software. We appreciate your business.
```

We see that the map induced on homology is what we expect: ``H_2 (P_1,\ P_0) \cong \mathbb{Z}``, ``H_2 (Q_1,\ Q_0) \cong \mathbb{Z}`` and ``f_*`` sends the generator of ``H_2 (P_1,\ P_0)`` to the generator of ``H_2 (Q_1,\ Q_0)``. 
