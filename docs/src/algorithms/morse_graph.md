# Morse Graphs

(TODO)

```@docs; canonical=false
morse_graph(F♯::TransferOperator)
```

### Example

```@repl 1
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

G = morse_graph(F♯)

labels(G)

G[1]

tiles = [G[label] for label in labels(G)]

using Plots

p = plot();

for (i, tile) in enumerate(tiles)
    p = plot!(p, tile, color=i)
end

savefig(p, "tiles.svg"); nothing # hide
```

![Morse sets](tiles.svg)
