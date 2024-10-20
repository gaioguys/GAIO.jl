using GAIO
using StaticArrays

const A = SA_F64[0.5 0;
                 0   2]

f(x) = A*x

c, r = (0,0), (4,4)
domain = Box(c, r)

P = BoxGrid(domain, (64,64))
S = cover(P, c)

F = BoxMap(:interval, f, domain)

N = isolating_neighborhood(F, S)
P1, P0, Q1, Q0 = index_quad(F, N)
transfers = TransferOperator(F, P1, Q1)

using Plots
pl = plot(P1)
pl = plot!(pl, P0, color=:blue)
pl = plot(Q1)
pl = plot!(pl, Q0, color=:blue)

@save P1
@save P0
@save Q1
@save Q0
@save transfers
