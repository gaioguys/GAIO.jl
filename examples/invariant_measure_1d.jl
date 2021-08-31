using GAIO

# logistic map
f(x) = 4.0.*x.*(1.0.-x)

center, radius = 0.5, 0.5
Q = Box(center, radius)
depth = 8
P = RegularPartition(Q, depth)

F = BoxMap(f, Q; no_of_points=400)

T = TransferOperator(F, P[:])
(λ, ev) = eigs(T)

plot(abs ∘ ev[1])
