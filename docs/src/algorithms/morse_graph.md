# Morse Graphs

(TODO)

### Example

```@example 1
using GAIO 
using Graphs, MetaGraphsNext

const θ₁, θ₂ = 20., 20.
f((x, y)) = ( (θ₁*x + θ₂*y) * exp(-0.1*(x + y)), 0.7*x )

center, radius = (38, 26), (38, 26)
Q = Box(center, radius)
P = BoxPartition(Q, (256,256))
S = cover(P, :)

F = BoxMap(:interval, f, Q)
F♯ = TransferOperator(F, S, S)
```

```@example 1
component_map = morse_component_map(F♯)
```

```@example 1
S = F♯.domain    # F♯.domain is ordered, it might differ in order from the original S
```

```@example 1
[key => component for (key, component) in zip(keys(S), component_map)]
```

```@example 1
adjacencies = morse_graph(F♯)
```

```@example 1
tiles = morse_tiles(F♯)
```

```@example 1
adjacencies, tiles = morse_graph_and_tiles(F♯)    # less computation than doing separately
```

```@example 1
G = MetaGraph(adjacencies, tiles)
```

```@example 1
labels(G)
```

```@example 1
G[1]
```

```@example 1
using Plots

p = plot(tiles, colormap=:jet)

savefig(p, "tiles.svg"); nothing # hide
```

![Morse sets](tiles.svg)
