# Morse Graphs

### Mathematical Background

A pivotal result in dynamical systems theory due to Conley states (in essence) that a dynamical system can be deconstructed into "recurrent" and "gradient-like" parts. The recurrent dynamics are captured within the recurrent sets, and are often analyzed using the [Conley Index](@ref). 

Much the same way that morse theory can describe a smooth closed manifold by means of a gradient field, Conley-Morse theory proves the existence of a gradient field which captures the dynamics of the nonrecurrent components of a dynamical system (this is referred to as Conley's decomposition theorem [conley](@cite), or occasionally the fundamental theorem of dynamical systems). 

For practical purposes, this means a dynamical system can be described as an acyclic directed graph where the nodes are the recurrent components, and edges are drawn if there exists an orbit between recurrent components. 

### Example

```@repl 1
using GAIO, MetaGraphsNext

const θ₁, θ₂ = 20., 20.
f((x, y)) = ( (θ₁*x + θ₂*y) * exp(-0.1*(x + y)), 0.7*x )

center, radius = (38, 26), (38, 26)
Q = Box(center, radius)
P = BoxGrid(Q, (256,256))
S = cover(P, :)

F = BoxMap(:interval, f, Q)
F♯ = TransferOperator(F, S, S)

G = morse_graph(F♯)

labels(G)

tiles = [G[label] for label in labels(G)]

using Graphs

n = nv(G) # number of vertices
```

### Plotting Morse Graphs with Plots.jl or Makie.jl

To plot a graph using Plots.jl, there is the package GraphRecipes.jl: 

```@example 1
using Plots
using GraphRecipes

colors = cgrad(palette(:auto, n), categorical=true)

p1 = graphplot(
    G.graph,
    names=1:n,
    markercolor=collect(colors),
    markersize=0.2
)

p2 = plot(BoxMeasure(G), colormap=colors, colorbar=false)

p = plot(p1, p2)

p = plot(p1, p2, dpi=500) # hide
savefig(p, "morse_graph.png"); nothing # hide
```

![Morse graph](morse_graph.png)

The same result can be created using Makie.jl and GraphMakie.jl: 

```julia
using GLMakie
using GraphMakie

colors = Makie.wong_colors()    # default colors
colors = colors[ [1:n;] .% length(colors) .+ 1 ]

fig, ax, ms = graphplot(
    G.graph,
    ilabels=1:n,
    node_color=colors
)
    
ax = Axis(fig[1,2])
for (i, label) in enumerate(labels(G))
    morse_set = G[label]
    plot!(ax, morse_set, color=colors[i])
end
```

```@docs; canonical=false
morse_graph
```
