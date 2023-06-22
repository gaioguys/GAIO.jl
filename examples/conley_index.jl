using GAIO
using SparseArrays
using OrderedCollections
using LinearAlgebra
using Plots

function GAIO.BoxSet(N::BoxSet, vals; settype=OrderedSet)
    BoxSet(N.partition, settype(key for (key,val) in zip(N.set, vals) if val != 0))
end

function neighborhood(B::BoxSet)
    P = B.partition
    nbhd = empty!(copy(B))
    for (c, r) in B
        box = Box(c, 1.5 .* r)
        union!(nbhd, cover(P, box))
    end
    return setdiff!(nbhd, B)
end

function neighborhood(B::BoxSet{R,Q}) where {N,R,Q<:BoxPartition{N}}
    surrounding = CartesianIndices(ntuple(_-> -1:1, N))
    nbhd = empty!(copy(B))
    for key in B.set
        union!(nbhd, (key .+ c.I for c in surrounding))
    end
    return setdiff!(nbhd, B)
end

const nbhd = neighborhood

function fast_positive_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁺, 1); fill(v⁻, 0)
    M = F♯.mat
    N = F♯.domain
    while v⁺ != v⁻
        v⁻ .= v⁺
        v⁺ .= M*v⁻
        v⁺ .= min.(v⁺, v⁻)
    end
    S⁺ = BoxSet(N, v⁺)
end

function fast_negative_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁻, 1); fill!(v⁺, 0)
    M = F♯.mat
    N = F♯.domain
    while v⁻ != v⁺
        v⁺ .= v⁻
        v⁻ .* M'v⁺
        v⁻ .= min.(v⁺, v⁻)
    end
    S⁻ = BoxSet(N, v⁻)
end

function index_pair(F::BoxMap, N::BoxSet)
    N = N ∪ nbhd(N)

    F♯ = TransferOperator(F, N, N)
    v⁺ = Vector{Float64}(undef, length(N))
    v⁻ = Vector{Float64}(undef, length(N))

    S⁺ = fast_positive_invariant_part(F♯, v⁺, v⁻)
    S⁻ = fast_negative_invariant_part(F♯, v⁺, v⁻)

    P₁ = S⁻
    P₀ = setdiff(S⁻, S⁺)
    return P₁, P₀
end

function index_quad(F::BoxMap, N::BoxSet)
    P₁, P₀ = index_pair(F, N)
    FP₁ = F(P₁)
    P̄₁ = P₁ ∪ FP₁
    P̄₀ = P₀ ∪ setdiff(FP₁, P₁)
    return P₁, P₀, P̄₁, P̄₀
end

function period_n_orbit(F, N; n=2, settype=OrderedSet)
    F♯ = TransferOperator(F, N, N)
    M = sparse(F♯)
    N = F♯.domain

    for _ in 2:n
        M .= F♯.mat * M
    end

    v = diag(M)
    BoxSet(N, v .> 0; settype=settype)
end

# the Henon map
const a, b = 1.4, 0.2
f((x,y)) = (1 - a*x^2 + y/5, 5*b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(:adaptive, f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 14)

per = [period_n_orbit(F, A; n=n) for n in 1:6]
B = union(per[2:end]...)
B = setdiff!(B, per[1])

P₁, P₀ = index_pair(F, B)
P₁, P₀, P̄₁, P̄₀ = index_quad(F, B)

p = plot(A, alpha=0.4, size=(900,600))
p = plot!(p, B, color=:blue)
p = plot!(p, P₁, color=:green)
p = plot!(p, P₀, color=:darkblue)
p = plot!(p, P̄₁, color=:orange)
p = plot!(p, P̄₀, color=:pink)
