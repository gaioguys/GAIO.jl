"""
    Graph(gstar::TransferOperator) -> BoxGraph
    Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset, boxset))

Directed Graph representation of a `TransferOperator`. The 
boxes in a `BoxSet` are enumerated as in TransferOperator. 
This means if the `domain`, `codomain` are taken from a 
`TranferOperator`, then the graph vertices are numbered 
`1 .. length(domain ∪ codomain)`. 

A directed edge exists from the i'th box `b_i` to the j'th 
box `b_j` if the BoxMap `g` has `b_j ∩ g⁻¹(b_i) ≠ ∅`. 
Equivalently, 
```julia
has_edge(g::BoxGraph, i, j) = !iszero( Matrix(g.gstar)[j,i] )
```

`Graphs.jl` operations like 
```julia
vertices, edges, weights, inneighbors, outneighbors, # etc...
```
are supported. Algorithms in `Graphs.jl` 
should work "out of the box", but will return whatever `Graphs.jl` 
returns by default. To convert a (integer) vertex index from the 
graph into a box index from the partition, one can call 
```julia
BoxSet(boxgraph, graph_index_or_indices)
```
If you would like to see specific behavior 
implemented, please open an issue! 

## Implementation details, not important for use:

We want to turn a matrix representation 
```julia
        domain -->
codomain  .   .   .   .   .
    |     .   .   .   .   .
    |     .   .   .   .   .
    v     .   .  mat  .   .
          .   .   .   .   .
          .   .   .   .   .
```
into a graph representation 
```julia
  domain ∪ codomain
  .---------.   .
 / \\       /   /
.   .-----.---.
```
!! efficiently !!

Julia's Graphs package only allows integer-indexed
vertices so we need to enumerate domain ∪ codomain. 
To do this, we enumerate the domain, then skip 
the boxes in the codomain which are already in the 
domain, then continue enumerating the rest of the 
codomain. 

We therefore permute the row indices of the weight 
matrix so that the skipped elements of the codomain
come first. 
"""
struct BoxGraph{B,T,P<:TransferOperator{B,T}} <: Graphs.AbstractSimpleGraph{Int}
    gstar::P
    n_intersections::Int
end

function BoxGraph(gstar::TransferOperator)
    gstar.codomain === gstar.domain && return BoxGraph(gstar, length(gstar.domain))
    # permute the row indices so that we can skip already identified boxes
    cut = gstar.codomain ∩ gstar.domain
    n = length(cut)
    inds = [getkeyindex(gstar.codomain, key) for key in cut.set]
    gstar.mat[[1:n; inds], :] .= gstar.mat[[inds; 1:n], :]
    return BoxGraph(gstar, n)
end

Graphs.Graph(gstar::TransferOperator) = BoxGraph(gstar)
Graphs.Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset, boxset))

function Base.show(io::IO, g::BoxGraph)
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of $(g.gstar)")
end

Base.eltype(::BoxGraph{B,T}) where {B,T} = B
Graphs.edgetype(::BoxGraph) = Graphs.SimpleEdge{Int}
Graphs.is_directed(::Type{<:BoxGraph}) = true
Graphs.is_directed(::BoxGraph) = true
Graphs.weights(g::BoxGraph) = g.gstar.mat'

Graphs.nv(g::BoxGraph) = sum(size(g.gstar.mat)) - g.n_intersections
Graphs.vertices(g::BoxGraph) = 1:nv(g)
Graphs.has_vertex(g::BoxGraph, i) = 1 ≤ i ≤ nv(g)

Graphs.ne(g::BoxGraph) = length(nonzeros(g.gstar))

function Graphs.edges(g::BoxGraph)
    m, n = size(g.gstar)
    Iterators.map(zip(findnz(g.gstar.mat)...)) do nz
        i, ĵ, _ = nz
        if ĵ > g.n_intersections
            j = ĵ - n
        else
            u = getindex_fromkeys(g.gstar.codomain, ĵ)
            j = getkeyindex(g.gstar.domain, u)
        end
        graphs.SimpleEdge{Int}(j, i)
    end
end

function Graphs.has_edge(g::BoxGraph, i, j)
    m, n = size(g.gstar)
    v = g.gstar.mat[i, j ≤ n ? j : j - n + g.n_intersections]
    return !iszero(v)
end

Graphs.SimpleGraphs.badj(g::BoxGraph, v) = collect(inneighbors(g, v))
Graphs.SimpleGraphs.badj(g::BoxGraph) = [Graphs.SimpleGraphs.badj(g, v) for v in Graphs.vertices(g)]
Graphs.SimpleGraphs.fadj(g::BoxGraph, v) = collect(outneighbors(g, v))
Graphs.SimpleGraphs.fadj(g::BoxGraph) = [Graphs.SimpleGraphs.fadj(g, v) for v in Graphs.vertices(g)]

function Graphs.LinAlg.adjacency_matrix(g::BoxGraph{B,T}) where {B,T} 
    g.gstar.domain === g.gstar.codomain && return copy(g.gstar.mat')
    
    m, n = size(g.gstar)
    set = g.gstar.domain ∪ g.gstar.codomain
    N = length(set)
    mat = spzeros(T, N, N)
    mat[n + g.n_intersections + 1 : end, 1:n] .= g.gstar.mat[n_intersections + 1 : end, :]

    cut = g.gstar.codomain ∩ g.gstar.domain
    dom_inds = [getkeyindex(g.gstar.domain, key) for key in cut.set]
    codom_inds = [getkeyindex(g.gstar.codomain, key) for key in cut.set]
    mat[dom_inds, 1:n] .= g.gstar.mat[codom_inds, :]

    return mat'
end

#Graphs.outneighbors(g::BoxGraph, v::Integer) = findall(!iszero, g.gstar.mat[:, v])
# efficiently find the nonzero rows corresponding to a column
function Graphs.outneighbors(g::BoxGraph, v::Integer)
    m, n = size(g.gstar)
    rows = rowvals(g.gstar.mat)
    # take nzrange for column or empty range if v > n, i.e. transfers out of v not calulated.
    # we do it this way to ensure that the result is type stable
    iterrange = 1 ≤ v ≤ n ? nzrange(g.gstar.mat, v) : (1:0)
    Iterators.map(@view rows[iterrange]) do ĵ
        if ĵ > g.n_intersections
            j = ĵ - n
        else
            u = getindex_fromkeys(g.gstar.codomain, ĵ)
            j = getkeyindex(g.gstar.domain, u)
        end
        j
    end
end

#Graphs.inneighbors(g::BoxGraph,  u::Integer) = findall(!iszero, g.gstar.mat[u, :])
# efficiently find the nonzero columns related to a row 
function Graphs.inneighbors(g::BoxGraph, u::Integer)
    m , n = size(g.gstar)
    j = u ≤ n ? u : u - n + g.n_intersections
    rows = rowvals(g.gstar.mat)
    colptr = SparseArrays.getcolptr(g.gstar.mat)
    (findfirst(>(i), colptr) - 1 for (i, row) in enumerate(rows) if row == j)
end

"""
    BoxSet(boxgraph, graph_index_or_indices) -> BoxSet

Construct a BoxSet from some 
index / indices of vertices in a BoxGraph. 
"""
function BoxSet(g::BoxGraph{P}, inds) where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    keys = map(inds) do j
        if j ≤ n
            getindex_fromkeys(g.gstar.domain, j)
        else
            getindex_fromkeys(g.gstar.codomain, j - n + g.n_intersections)
        end
    end

    return BoxSet(g.gstar.domain.partition, R(keys))
end

function BoxSet(g::BoxGraph{P}, ind::Integer) where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    BoxSet(g, (ind,))
end

function Graphs.SimpleDiGraph(gstar::TransferOperator)
    G = Graph(gstar)
    adj = adjacency_matrix(G)
    SimpleDiGraph(adj)
end

Graphs.SimpleDiGraph(g::BoxGraph) = SimpleDiGraph(g.gstar)
