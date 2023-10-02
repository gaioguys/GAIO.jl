using GAIO
using GLMakie
using ProgressMeter
const Box = GAIO.Box

b = 0.25

v((x, y, z)) = @. sin((y, z, x)) - b*(x, y, z)
f(x) = rk4_flow_map(v, x, 0.01, 40)

c, r = (0,0,0), (6,6,6)
domain = Box(c, r)

P = BoxPartition(domain, (256,256,256))
F = BoxMap(f, domain)

A = cover(P, c)
A = A ∪ nbhd(A)
B = copy(A)

while !isempty(B)
    B = F(B; show_progress=true)
    @show B = setdiff!(B, A)
    @show A = union!(A, B)
end

A = unstable_set(F, A)
A = BoxSet(morse_tiles(F, A))
plot(A)

F♯ = TransferOperator(F, A, A, show_progress=true)
λ, ev, nconv = eigs(F♯, which=:LR, nev=1, tol=eps()^(1/4), v0=ones(length(A)))

μ = log ∘ abs ∘ ev[1]
plot(μ, colormap=(:brg, 0.3))

