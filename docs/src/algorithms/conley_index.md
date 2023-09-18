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

const θ₁, θ₂ = 20., 20.
f((x, y)) = ( (θ₁*x + θ₂*y) * exp(-0.1*(x + y)), 0.7*x )

center, radius = (38, 26), (38, 26)
Q = Box(center, radius)
P = BoxPartition(Q, (256,256))
S = cover(P, :)

F = BoxMap(:interval, f, Q)
F♯ = TransferOperator(F, S, S)

adjacencies, tiles = morse_graph_and_tiles(F♯)    # less computation than doing separately
```

```@example 1
B = BoxSet(P, Set(key for (key,val) in tiles.vals if val==3))   # choose an arbitrary tile
```

```@example 1
P1, P0 = index_pair(F, B)
```

```@example 1
P1, P0, Q1, Q0 = index_quad(F, B)
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
# get `homcubes` at http://www.pawelpilarczyk.com/chomp/software/
# run(`/home/april/Downloads/chomp_full/bin/homcubes -i -g generators.dat transfers.map P1.cub P0.cub Q1.cub Q0.cub`)
pwd()
```
