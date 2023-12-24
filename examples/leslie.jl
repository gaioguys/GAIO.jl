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

n = nv(G) # number of vertices


# --- choose either Plots or Makie ---

using Plots
using GraphRecipes

colors = cgrad(palette(:auto, n), categorical=true)

p1 = graphplot(
    G.graph,
    names=1:n,
    markercolor=collect(colors),
    markersize=0.2
);

p2 = plot(BoxFun(G), colormap=colors, colorbar=false);

fig = plot(p1, p2)

# ------------------------------------

using GLMakie
using GraphMakie

colors = Makie.wong_colors()
colors = colors[ [1:n;] .% length(colors) .+ 1 ]

fig, ax, ms = graphplot(
    G.graph,
    ilabels=1:n,
    node_color=colors
);
    
ax = Axis(fig[1,2])
for (i, label) in enumerate(labels(G))
    morse_set = G[label]
    plot!(ax, morse_set, color=colors[i])
end

fig

# ------------------------------------
