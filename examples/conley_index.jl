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

    return setdiff!(C, B)
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


# ------------------------------------------------------------------------------------


using CHomP
using GAIO
using IntervalArithmetic, StaticArrays, Chain
#ENV["JULIA_DEBUG"] = all


low(int::IntervalBox) = getfield.(int.v, :lo)
low(box::Box) = box.center .- box.radius
high(int::IntervalBox) = getfield.(int.v, :hi)
high(box::Box) = box.center .+ box.radius

function evaluate!(f, box_vec::Array{T}) where {T}
    N = length(box_vec) ÷ 2
    lo_hi = reinterpret(SVector{N,T}, box_vec)
    int = IntervalBox(lo_hi...)
    int = IntervalBox(f(int))
    lo_hi[1] = low(int)
    lo_hi[2] = high(int)
    return box_vec
end

size_and_type(::Box{N,T}) where {N,T} = (N, T)

function morse_set(partition::P, morse_graph, vertex) where {N,T,P<:AbstractBoxPartition{Box{N,T}}}
    @chain vertex begin
        morse_graph.morse_set_boxes(_)
        PermutedDimsArray(_, (2,1))
        reinterpret(SVector{N,T}, _)
        eachcol(_)
        IntervalBox.(_)
        cover(partition, _)
    end
end

function conley_morse_graph(F::BoxMap, depth)
    domain = F.domain
    𝓕(box_vec) = evaluate!(F.map, box_vec)

    model = cmgdb.Model(depth, depth, low(F.domain), high(F.domain), 𝓕)
    morse_graph, map_graph = cmgdb.ComputeConleyMorseGraph(model)

    N, T = size_and_type(domain)
    P = BoxPartition(TreePartition(domain), depth)

    vertices_list = morse_graph.vertices()
    edges_list = morse_graph.edges()

    morse_sets = [morse_set(P, morse_graph, vert) for vert in vertices_list]

    pyconleyindex = pybuiltin("getattr")(morse_graph, "annotations", [])
    conley_indices = reshape(
        [label for vert in vertices_list for label in pyconleyindex(vert)],
        N+1, :
    )

    return vertices_list, edges_list, morse_sets, conley_indices
end



# Leslie map
const th1 = 19.6
const th2 = 23.68

f((x, y)) = ( (th1*x + th2*y) * exp(-0.1*(x + y)), 0.7*x )

depth = 21
lower_bounds = [-0.001, -0.001]
upper_bounds = [90., 70.]

domain = Box(IntervalBox(lower_bounds, upper_bounds))
P = BoxPartition(TreePartition(domain), depth)

F = BoxMap(f, domain)
vertices_list, edges_list, morse_sets, conley_indices = conley_morse_graph(F, depth)



using DataStructures
using SparseArrays
using MatrixNetworks, Graphs
adj = sparse([
    0 1 0 0 0 0 0 0;
    0 0 1 1 0 0 0 0;
    1 0 0 0 1 0 0 0;
    0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 1;
    0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 1 0;
])

G = MatrixNetwork(adj)
scomp = scomponents(G)
scomp_enr = enrich(scomp)
fieldnames(typeof(scomp))
fieldnames(typeof(scomp_enr))


Graphs.vertices(mat::AbstractSparseMatrix) = 1:size(mat, 1)
Graphs.inneighbors(mat::AbstractSparseMatrix, v) = checkbounds(Bool, mat, :) ? findall(!iszero, mat[:, v]) : Int64[]
Graphs.outneighbors(mat::AbstractSparseMatrix, u) = checkbounds(Bool, mat, :) ? findall(!iszero, mat[u, :]) : Int64[]
Graphs.has_edge(mat::AbstractSparseMatrix, u, v) = checkbounds(Bool, mat, u, v) && !iszero(mat[u, v])
Graphs.add_edge!(mat::AbstractSparseMatrix, u, v) = checkbounds(Bool, mat, u, v) && (mat[u, v] = 1; true)
#Graphs.rem_vertex!(mat::AbstractSparseMatrix, v) = (n = size(mat, 1); vs = [1:v-1; v+1:n]; mat[vs, vs])
#Graphs.rem_vertices!(mat::AbstractSparseMatrix, vs) = (n = size(mat, 1); ws = setdiff(1:n, vs); mat[ws, ws])
isolate!(mat::AbstractSparseMatrix, v) = (mat[:, v] .= 0; mat[v, :] .= 0; mat)#; dropzeros!(mat))
is_isolated(mat::AbstractSparseMatrix, v) = all(mat[:, v] .== 0) && all(mat[v, :] .== 0)

function rem_components!(morse_map, isolated)
    cumsum_isolated = cumsum(isolated)
    for v in eachindex(morse_map)
        morse_map[v] == -1 || (morse_map[v] -= cumsum_isolated[v])
    end
    return morse_map
end

function strong_components(mat)
    G = MatrixNetwork(mat)

    scomp = scomponents(G)
    sizes, comp_map = scomp.sizes, scomp.map

    scomp_enr = enrich(scomp)
    reduced_adj = scomp_enr.transitive_map

    return comp_map, sizes, reduced_adj
end

"""
construct condensation graph and morse graph
"""
function reduce(mat::AbstractSparseMatrix)
    comp_map, sizes, reduced_adj = strong_components(mat)
    condensation_graph = copy(reduced_adj)

    V = vertices(reduced_adj)
    morse_map, to_search = collect(V), collect(V)
    
    while !isempty(to_search)
        v = pop!(to_search)

        if sizes[v] == 1 && !has_edge(reduced_adj, v, v)
            for u in inneighbors(reduced_adj, v), w in outneighbors(reduced_adj, v)
                add_edge!(reduced_adj, u, w)
            end
            isolate!(reduced_adj, v)
            morse_map[v] = -1    # arbitrary: gradient-like portion gets ID -1
        end
    end

    isolated = is_isolated.(Ref(reduced_adj), V)
    morse_map = rem_components!(morse_map, isolated)
    morse_graph = reduced_adj[.!isolated, .!isolated]

    return condensation_graph, comp_map, morse_graph, morse_map
end

function generate_search(adj)
    to_search = Deque{Int}()
    search_set = Set{Int}()
    for v in vertices(adj)
        push!(to_search, v)
        push!(search_set, v)
    end
    return to_search, search_set
end

"""
Given an acyclic directed graph representing a poset (P, ≼), 
construct the graph which has an edge 
(u,v) ∈ P×P  iff  u ≼ v
"""
function comparability_graph(mat::AbstractSparseMatrix)
    adj = copy(mat)
    to_search, search_set = generate_search(adj)
    
    while !isempty(search_set)
        v = pop!(to_search)
        delete!(search_set, v)

        for w in outneighbors(adj, v)
            if w in search_set
                pushfirst!(to_search, v)
                push!(search_set, v)
                break
            end
            for u in inneighbors(adj, v)
                add_edge!(adj, u, w)
            end
        end
    end

    return adj
end

function morse_component_map(component_map, morse_map)
    [morse_map[component_map[v]] for v in eachindex(component_map)]
end

function regions_of_attraction(
        condensation_graph, 
        morse_comparability_graph, 
        morse_map
    )

    region_of_attraction = copy(morse_map)
    to_search = PriorityQueue(Base.Order.Reverse, enumerate(region_of_attraction))
    
    while !isempty(to_search)
        v, prio = dequeue_pair!(to_search)
        is_rooted = true

        for w in outneighbors(condensation_graph, v)
            if haskey(to_search, w)
                is_rooted = false
                to_search[w] += 1
            end
        end

        if is_rooted
            roa_v = region_of_attraction[v]

            for u in inneighbors(condensation_graph, v)
                roa_u = region_of_attraction[u]

                if roa_u == -1 || has_edge(morse_comparability_graph, u, v)
                    region_of_attraction[u] = roa_v
                end
            end
        else
            to_search[v] = prio - 1
        end
    end

    return region_of_attraction
end

function regions_of_attraction(roas, component_map)
    [roas[component_map[v]] for v in eachindex(component_map)]
end
