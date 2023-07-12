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
    P = B.partition
    C = empty!(copy(B))

    surrounding = CartesianIndices(ntuple(_-> -1:1, N))
    nbhd(key) = Iterators.filter(
        x -> checkbounds(Bool, P, x),
        (key .+ Tuple(cartesian_ind) for cartesian_ind in surrounding)
    )

    for key in B.set
        union!(C, nbhd(key))
    end

    return setdiff!(nbhd, B)
end

const nbhd = neighborhood

⊓(a, b) = a > 0 && b > 0

function fast_positive_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁺, 1); fill(v⁻, 0)
    M = F♯.mat
    while v⁺ != v⁻
        v⁻ .= v⁺
        v⁺ .= M*v⁻
        v⁺ .= v⁺ .⊓ v⁻
    end
    S⁺ = BoxSet(F♯.domain, v⁺)
end

function fast_negative_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁻, 1); fill!(v⁺, 0)
    M = F♯.mat
    while v⁻ != v⁺
        v⁺ .= v⁻
        v⁻ .* M'v⁺
        v⁻ .= v⁺ .⊓ v⁻
    end
    S⁻ = BoxSet(F♯.domain, v⁻)
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

function matching_partitions(S1::BoxSet, S2::BoxSet)
    P, Q = S1.partition, S2.partition
    P == Q
end

for (type, func) in Dict(
        TransferOperator                                    => identity,
        LinearAlgebra.Transpose{<:Any,<:TransferOperator}   => transpose,
        LinearAlgebra.Adjoint{<:Any,<:TransferOperator}     => adjoint
    )

    @eval function Base.:(*)(F♯::$type, S::BoxSet)
        dom = F♯.domain
        supp = matching_partitions(dom, S) ? S : cover(dom, S)
        μ = BoxFun(supp)

        T = $func(F♯)
        μ = T * μ

        return BoxSet(μ)
    end

end

function positive_invariant_part(F♯)
    S₁ = F♯.domain
    S₂ = empty!(copy(S₁))
    while S₁ != S₂
        S₂ = copy(S₁)
        S₁ = F♯*S₁ ∩ S₁
    end
    return S₁
end

function negative_invariant_part(F♯)
    S₁ = F♯.domain
    S₂ = empty!(copy(S₁))
    while S₁ != S₂
        S₂ = copy(S₁)
        S₁ = F♯'S₁ ∩ S₂
    end
    return S₁
end


struct ModuloBoxGraph{S<:AbstractSet}
    graph::BoxGraph
    mod_components::Vector{S}
end

function component_map(g::ModuloBoxGraph, v)
    i = findfirst(comp -> v ∈ comp, g.mod_components)
    isnothing(i) ? v : nv(g.graph) + i
end

function inverse_component_map(g::ModuloBoxGraph, v)
    v ≤ nv(g.graph) ? [v] : g.mod_components[v]
end

Graphs.is_directed(::Type{<:ModuloBoxGraph}) = true
Graphs.is_directed(::ModuloBoxGraph) = true
Graphs.edgetype(::ModuloBoxGraph) = Graphs.Edge{Int}
Graphs.nv(g::ModuloBoxGraph) = nv(g.graph)+length(g.mod_components)
Graphs.vertices(g::ModuloBoxGraph) = map(v -> component_map(g, v), vertices(g.graph))
Graphs.has_vertex(g::ModuloBoxGraph, v) = v == inverse_component_map(g, v)[1]
Graph.edges(g::ModuloBoxGraph) = unique(
    Graphs.Edge{Int}(component_map(g, u), component_map(g, v))
    for (u, v) in edges(g.graph)
)

function Graphs.has_edge(g::ModuloBoxGraph, u, v)
    w = inverse_component_map(g, u)
    y = inverse_component_map(g, v)
    for c in CartesianIndices((eachindex(w), eachindex(y)))
        i, j = c.I
        has_edge(g.graph, w[i], v[j]) && return true
    end
    return false
end

function Graphs.outneighbors(g::ModuloBoxGraph, v)
    all_nbrs = (
        map(i -> component_map(g, i), outneighbors(g.graph, j)) 
        for j in inverse_component_map(g, v)
    )
    union(all_nbrs...)
end

function Graphs.inneighbors(g::ModuloBoxGraph, v)
    all_nbrs = (
        map(i -> component_map(g, i), inneighbors(g.graph, j)) 
        for j in inverse_component_map(g, v)
    )
    union(all_nbrs...)
end



