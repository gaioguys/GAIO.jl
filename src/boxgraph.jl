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
    cut = intersect!(copy(gstar.domain), gstar.codomain)
    gstar.codomain = union!(cut, gstar.codomain)
    
    inds = [key_to_index(gstar.codomain, key) for key in cut.set]
    reorder_rows!(gstar.mat, inds)
    return BoxGraph(gstar, n)
end

Graphs.Graph(gstar::TransferOperator) = BoxGraph(gstar)
Graphs.Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset, boxset))

function Base.show(io::IO, g::BoxGraph{B,T,P}) where {B,T,P}
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of TransferOperator")
end

function Base.show(io::IO, ::MIME"text/plain", g::BoxGraph{B,T,P}) where {B,T,P}
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of TransferOperator")
end

function row_to_index(g::BoxGraph, row)
    g.gstar.domain === g.gstar.codomain && return row
    if row ≤ g.n_intersections
        u = index_to_key(g.gstar.codomain, row)
        j = key_to_index(g.gstar.domain, u)    
    else
        m, n = size(g.gstar)
        j = row + n - g.n_intersections
    end
    return j
end

function index_to_row(g::BoxGraph, j)
    g.gstar.domain === g.gstar.codomain && return j
    m, n = size(g.gstar)
    if j ≤ n
        u = index_to_key(g.gstar.domain, j)
        row = key_to_index(g.gstar.codomain, u)
    else
        row = j - n + g.n_intersections
    end
    return row
end

# partition-key to vertex-index
function key_to_index(g::BoxGraph, u)
    i = key_to_index(g.gstar.domain, u)
    !isnothing(i) && return i
    j = row_to_index(g, key_to_index(g.gstar.codomain, u))
    return j
end

# vertex-index to partition-key
function index_to_key(g::BoxGraph, j)
    m, n = size(g.gstar)
    j ≤ n && return index_to_key(g.gstar.domain, j)
    return index_to_key(g.gstar.codomain, index_to_row(g, j))
end

Base.eltype(::BoxGraph{B,T}) where {B,T} = B
Graphs.edgetype(::BoxGraph) = Graphs.SimpleEdge{Int}
Graphs.is_directed(::Type{<:BoxGraph}) = true
Graphs.is_directed(::BoxGraph) = true
Graphs.weights(g::BoxGraph) = g.gstar.mat'

Graphs.nv(g::BoxGraph) = sum(size(g.gstar.mat)) - g.n_intersections
Graphs.vertices(g::BoxGraph) = 1:nv(g)
Graphs.has_vertex(g::BoxGraph, i::Integer) = 1 ≤ i ≤ nv(g)
Graphs.has_vertex(g::BoxGraph, key) = has_vertex(g, key_to_index(g, key))

function Graphs.edges(g::BoxGraph)
    return (
        Graphs.SimpleEdge{Int}(i, row_to_index(g, j))
        for (j,i,w) in zip(findnz(g.gstar.mat)...)
    )
end
        
function Graphs.has_edge(g::BoxGraph, i::Integer, j::Integer)
    ĵ = index_to_row(g, j)
    v = g.gstar.mat[ĵ, i]
    return !iszero(v)
end

Graphs.ne(g::BoxGraph) = length(nonzeros(g.gstar.mat))
Graphs.has_edge(g::BoxGraph, u, v) = has_edge(g, key_to_index(g, u), key_to_index(g, v))

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
    mat[n + g.n_intersections + 1 : end, 1:n] .= g.gstar.mat[g.n_intersections + 1 : end, :]

    cut = g.gstar.codomain ∩ g.gstar.domain
    dom_inds = [key_to_index(g.gstar.domain, key) for key in cut.set]
    codom_inds = [key_to_index(g.gstar.codomain, key) for key in cut.set]
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

    return [ row_to_index(g, row) for row in @view(rows[iterrange]) ]
end

#Graphs.inneighbors(g::BoxGraph,  u::Integer) = findall(!iszero, g.gstar.mat[u, :])
# efficiently find the nonzero columns related to a row 
function Graphs.inneighbors(g::BoxGraph, u::Integer)
    rows = rowvals(g.gstar.mat)
    colptr = SparseArrays.getcolptr(g.gstar.mat)
    j = index_to_row(g, u)
    return [ findfirst(>(i), colptr) - 1 for (i, row) in enumerate(rows) if row == j ]
end

function union_strongly_connected_components(g::BoxGraph)
    P = g.gstar.domain.partition

    sccs = Graphs.strongly_connected_components_tarjan(Graphs.IsDirected{typeof(g)}, g)
    connected_vertices = OrderedSet{keytype(typeof(P))}()

    for scc in sccs
        if length(scc) > 1 || has_edge(g, scc[1], scc[1])
            union!(
                connected_vertices, 
                (index_to_key(g, i) for i in scc)
            )
        end
    end
    
    return BoxSet(P, connected_vertices)
end

"""
    BoxSet(boxgraph, graph_index_or_indices) -> BoxSet

Construct a BoxSet from some 
index / indices of vertices in a BoxGraph. 
"""
function BoxSet(g::BoxGraph{B,T,P}, inds) where {B,T,B1,P1,R,S<:BoxSet{B1,P1,R},P<:TransferOperator{B,T,S}} # where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    keys = (index_to_key(g, i) for i in inds)
    return BoxSet(g.gstar.domain.partition, R(keys))
end

function BoxSet(g::BoxGraph{B,T,P}, ind::Integer) where {B,T,B1,P1,R,S<:BoxSet{B1,P1,R},P<:TransferOperator{B,T,S}} # where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    BoxSet(g, (ind,))
end

function Graphs.SimpleDiGraph(gstar::TransferOperator)
    G = Graph(gstar)
    adj = adjacency_matrix(G)
    SimpleDiGraph(adj)
end

Graphs.SimpleDiGraph(g::BoxGraph) = SimpleDiGraph(g.gstar)

# helper function to reorder rows in sparse matrix such that `inds` come first
function reorder_rows!(mat, inds)
    order = [ axes(mat, 1); ]
    order = [ order[inds]; order[Not(inds)] ]
    
    rows = rowvals(mat)
    for i in eachindex(rows)
        rows[i] = rows[order[i]]
    end

    return mat
end
