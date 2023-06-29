using GAIO

# Devil's curve
const a, b = 0.9, 1.0
H((x, y)) = x^2 * (x^2 - b^2) - y^2 * (y^2 - a^2)

domain = Box((0,0), (2,2))
P = BoxPartition(domain)
S = cover(P, :)

M = cover_manifold(H, S; steps=16)

using Plots: plot
#using WGLMakie

plot(M)
