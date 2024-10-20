using SnoopCompile, ProfileView

#invalidations = @snoopr begin

using GAIO

c = r = (0.5,)
Q = Box(c,r)
P = BoxGrid(Q, (10,))
S = cover(P, (0,))

f(x) = (x+0.5) % 1
fr(x) = f.(x)
F = BoxMap(fr, Q)
#C = F(S)

#end

#trees = SnoopCompile.invalidation_trees(invalidations)

tinf = @snoopi_deep F(S)
ProfileView.view(flamegraph(tinf))

tinf = @snoopi_deep TransferOperator(F, S)
ProfileView.view(flamegraph(tinf))
