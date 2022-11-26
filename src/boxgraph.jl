"""
    Graph(gstar::TransferOperator) -> BoxGraph
    Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset))

Directed Graph representation of a TransferOperator. The 
boxes in a BoxSet are enumerated as in TransferOperator. 
This means the vertices are numbered `1 .. size(gstar)[1]`. 
A directed edge exists from the i'th box `b_i` to the j'th 
box `b_j` in `boxgraph.gstar.support` if the BoxMap `g` 
has `b_j ∩ g⁻¹(b_i) ≠ ∅`. Equivalently, 
```julia
has_edge(g::BoxGraph, i, j) = !iszero( Matrix(g.gstar)[j,i] )
```

`Graphs.jl` operations like 
```julia
vertices, edges, weights, inneighbors, outneighbors, # etc...
```
are supported. Some algorithms are adapted like 
```julia
strongly_connected_components
```
to return custom objects. All other algorightms in `Graphs.jl` 
should work "out of the box", but will return whatever `Graphs.jl` 
returns by default. To convert a (integer) vertex index from the 
graph into a box index from the partition, one can call 
```julia
BoxSet(boxgraph, graph_index_or_indices)
```
If you would like to see specific behavior 
implemented, please open an issue! 

.
"""
struct BoxGraph{B,T,P<:TransferOperator{B,T}} <: Graphs.AbstractSimpleGraph{Int}
    gstar::P
end

Graphs.Graph(gstar::TransferOperator) = BoxGraph(gstar)
Graphs.Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset))

function Base.show(io::IO, g::BoxGraph)
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of $(g.gstar)")
end

Base.eltype(::BoxGraph{B,T}) where {B,T} = B
Graphs.edgetype(::BoxGraph) = Graphs.SimpleEdge{Int}
Graphs.is_directed(::Type{<:BoxGraph}) = true
Graphs.is_directed(::BoxGraph) = true
Graphs.weights(g::BoxGraph) = g.gstar.mat'

Graphs.nv(g::BoxGraph) = size(g.gstar)[1]
Graphs.vertices(g::BoxGraph) = 1:nv(g)
Graphs.has_vertex(g::BoxGraph, i) = 1 ≤ i ≤ nv(g)

Graphs.ne(g::BoxGraph) = length(collect(edges(g)))
Graphs.edges(g::BoxGraph) = (Graphs.SimpleEdge{Int}(j,i) for (i,j,w) in zip(findnz(g.gstar.mat)...))
Graphs.has_edge(g::BoxGraph, i, j) = checkbounds(Bool, g.gstar.mat', i, j) && !iszero(g.gstar.mat[j,i])

Graphs.SimpleGraphs.badj(g::BoxGraph, v) = collect(inneighbors(g, v))
Graphs.SimpleGraphs.badj(g::BoxGraph) = [Graphs.SimpleGraphs.badj(g, v) for v in Graphs.vertices(g)]
Graphs.SimpleGraphs.fadj(g::BoxGraph, v) = collect(outneighbors(g, v))
Graphs.SimpleGraphs.fadj(g::BoxGraph) = [Graphs.SimpleGraphs.fadj(g, v) for v in Graphs.vertices(g)]

function Graphs.LinAlg.adjacency_matrix(g::BoxGraph, T=Int)
    w = copy(weights(g))
    SparseMatrixCSC{T,Int}(
        size(w)..., 
        SparseArrays.getcolptr(w), 
        rowvals(w), 
        ones(T, length(nonzeros(w)))
    )
end

#Graphs.outneighbors(g::BoxGraph, v::Integer) = findall(!iszero, g.gstar.mat[:, v])
# efficiently find the nonzero rows corresponding to a column
function Graphs.outneighbors(g::BoxGraph, v::Integer)
    m, n = size(g.gstar)
    rows = rowvals(g.gstar.mat)
    # take nzrange for column or empty range if v > n, i.e. transfers out of v not calulated.
    # we do it this way to ensure that the result is type stable
    iterrange = 1 ≤ v ≤ n ? nzrange(g.gstar.mat, v) : (1:0)
    @view rows[iterrange]
end

#Graphs.inneighbors(g::BoxGraph,  u::Integer) = findall(!iszero, g.gstar.mat[u, :])
# efficiently find the nonzero columns related to a row 
function Graphs.inneighbors(g::BoxGraph, u::Integer)
    rows = rowvals(g.gstar.mat)
    colptr = SparseArrays.getcolptr(g.gstar.mat)
    (findfirst(>(i), colptr) - 1 for (i, row) in enumerate(rows) if row == u)
end

function Graphs.strongly_connected_components(g::BoxGraph)

    sccs = Graphs.strongly_connected_components(Graphs.IsDirected{typeof(g)}, g)
    connected_vertices = OrderedSet{keytype(typeof(g.gstar.support.partition))}()

    for scc in sccs
        if length(scc) > 1 || has_edge(g, scc[1], scc[1])
            union!(connected_vertices, map(j->getindex_fromkeys(g.gstar,j), scc))
        end
    end
    
    BoxSet(g.gstar.support.partition, connected_vertices)
end

"""
    BoxSet(boxgraph, graph_index_or_indices) -> BoxSet

Construct a BoxSet from some 
index / indices of vertices in a BoxGraph. 
"""
function BoxSet(g::BoxGraph{P}, inds) where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    BoxSet(
        g.gstar.support.partition, 
        R(getindex_fromkeys(g.gstar, j) for j in inds)
    )
end

function BoxSet(g::BoxGraph{P}, ind::Integer) where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    BoxSet(g, (ind,))
end

function Graphs.SimpleDiGraph(gstar::TransferOperator)
    SimpleDiGraphFromIterator(Graphs.SimpleEdge{Int}(j,i) for (i,j,w) in zip(findnz(gstar.mat)...))
end

Graphs.SimpleDiGraph(g::BoxGraph) = SimpleDiGraph(g.gstar)
