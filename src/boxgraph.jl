struct BoxGraph{B,T,P<:TransferOperator{B,T}} <: Graphs.AbstractSimpleGraph{Int}
    gstar::P
end

Graphs.Graph(gstar::TransferOperator) = BoxGraph(gstar)

function Base.show(io::IO, g::BoxGraph)
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of $(g.gstar)")
end

Base.eltype(::BoxGraph{B,T}) where {B,T} = B
Graphs.is_directed(::Type{<:BoxGraph}) = true
Graphs.is_directed(::BoxGraph) = true
Graphs.vertices(g::BoxGraph) = 1:nv(g)
Graphs.edges(g::BoxGraph) = (Graphs.SimpleEdge{Int}(j,i) for (i,j,w) in zip(findnz(g.gstar.mat)...) if i ≤ size(g.gstar)[2])
Graphs.ne(g::BoxGraph) = length(collect(edges(g)))
Graphs.nv(g::BoxGraph) = size(g.gstar)[2]
Graphs.edgetype(::BoxGraph) = Graphs.SimpleEdge{Int}
Graphs.has_edge(g::BoxGraph, i, j) = !iszero(g.gstar.mat[j,i]) # (all((j,i) .≤ size(g.gstar)) || return false; return )
Graphs.has_vertex(g::BoxGraph, i) = 1 ≤ i ≤ nv(g)
Graphs.SimpleGraphs.badj(g::BoxGraph, v) = collect(inneighbors(g, v))
Graphs.SimpleGraphs.badj(g::BoxGraph) = [Graphs.SimpleGraphs.badj(g, v) for v in Graphs.vertices(g)]
Graphs.SimpleGraphs.fadj(g::BoxGraph, v) = collect(outneighbors(g, v))
Graphs.SimpleGraphs.fadj(g::BoxGraph) = [Graphs.SimpleGraphs.fadj(g, v) for v in Graphs.vertices(g)]

function Graphs.outneighbors(g::BoxGraph, v::Integer)
    # efficiently find the nonzero rows corresponding to a column
    #v > size(g.gstar)[2] && return return []
    n = nv(g)
    rows = rowvals(g.gstar.mat)
    return (row for row in view(rows, nzrange(g.gstar.mat, v)) if row ≤ n)
end

function Graphs.inneighbors(g::BoxGraph, u::Integer)
    # efficiently find the nonzero columns related to a row 
    rows = rowvals(g.gstar.mat)
    colptr = SparseArrays.getcolptr(g.gstar.mat)
    (findfirst(>(i), colptr) - 1 for (i, row) in enumerate(rows) if row == u)
end

#Graphs.outneighbors(g::BoxGraph, v::Integer) = findall(!iszero, g.gstar.mat[1:size(g.gstar)[2], v])
#Graphs.inneighbors(g::BoxGraph,  u::Integer) = findall(!iszero, g.gstar.mat[u, 1:size(g.gstar)[2]])

function Graphs.strongly_connected_components(g::BoxGraph)

    sccs = Graphs.strongly_connected_components(Graphs.IsDirected{typeof(g)}, g)
    connected_vertices = OrderedSet{keytype(typeof(g.gstar.support.partition))}()

    for scc in sccs
        if length(scc) > 1 || has_edge(g, scc[1], scc[1])
            union!(connected_vertices, map(j->getindex_fromkeys(g.gstar,j), scc))
        end
    end
    
    return BoxSet(g.gstar.support.partition, connected_vertices)
end

function Graphs.SimpleDiGraph(gstar::TransferOperator)
    SimpleDiGraphFromIterator(Graphs.SimpleEdge{Int}(j,i) for (i,j,w) in zip(findnz(gstar.mat)...) if j ≤ size(gstar)[2])
end

Graphs.SimpleDiGraph(g::BoxGraph) = SimpleDiGraph(g.gstar)
