using GAIO
using WGLMakie
WGLMakie.activate!()

f(x) = 4.0.*x.*(1-x)
points = 2*rand(1000) .- 1
F = PointDiscretizedMap(f, points)

domain = Box((0.5), (0.5))
dep = 4
P = RegularPartition(domain, dep)
B = P[:]
Fstar = TransferOperator(F, B)
(λ, ev) = eigs(Fstar)

plot(abs ∘ ev[1]; color = :red)
