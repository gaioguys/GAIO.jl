using GAIO

center, radius = 0.5, 0.5
Q = Box(center, radius)

# logistic map
f(x) = 4.0.*x.*(1.0.-x)
F = BoxMap(f, Q; no_of_points=400)

depth = 8
P = RegularPartition(Q, depth)

T = TransferOperator(F, P[:])
(λ, ev) = eigs(T)

plot(abs ∘ ev[1]; color = :red)
