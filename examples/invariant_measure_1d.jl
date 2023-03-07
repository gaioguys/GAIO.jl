using GAIO

# logistic map
const μ = 4.0
f(x) = μ.*x.*(1.0.-x)

center, radius = 0.5, 0.5
P = BoxPartition(Box(center, radius), (256,))
F = BoxMap(:grid, f, P; no_of_points=(400,))

S = cover(P, :)
T = TransferOperator(F, S, S)
(λ, ev) = eigs(T)

using Plots: plot
#using WGLMakie: plot   # same result, just interactive

plot(abs ∘ ev[1])
