using GAIO

f(x) = 4.0.*x.*(1-x)
points = 2*rand(1000) .- 1
F = PointDiscretizedMap(f, points)

domain = Box((0.5), (0.5))
dep = 10
P = RegularPartition(domain, dep)
B = P[:]
T = TransferOperator(F, B)
(λ, ev) = eigs(T)

plot(abs ∘ ev[1]; ylim=(0,5))
