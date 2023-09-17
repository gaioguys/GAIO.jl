module MetaGraphsNextExt

using GAIO, MetaGraphsNext, SparseArrays, OrderedCollections
using MetaGraphsNext.Graphs: AbstractGraph, DiGraph, vertices, edges, src, dst

import MetaGraphsNext: MetaGraph
import GAIO: index_to_key, key_to_index, _rehash!

function MetaGraph(F♯::TransferOperator{B,T}) where {N,W,B<:Box{N,W},T}
    _rehash!(F♯)

    P = F♯.domain.partition
    adj = F♯.mat
    rows = rowvals(adj)
    vals = nonzeros(adj)

    G = MetaGraph(
        DiGraph();
        label_type = keytype(typeof(P)),
        edge_data_type = T,
        graph_data = P,
        weight_function = identity,
        default_weight = one(T)
    )

    for (col_j, key_j) in enumerate(F♯.domain.set)
        haskey(G, key_j) || (G[key_j] = nothing)

        for i in nzrange(adj, col_j)
            row_i = rows[i]
            key_i = index_to_key(F♯.codomain.set, row_i)
            weight = vals[i]

            haskey(G, key_i) || (G[key_i] = nothing)
            G[key_j, key_i] = weight
        end
    end
    
    return G
end

function MetaGraph(digraph::AbstractGraph{Code}, tiles::BoxFun{B,K,V}; settype=Set{K}) where {Code,B,K,V}
    P = tiles.partition
    edge_data = [(src(e), dst(e)) => nothing for e in edges(digraph)]

    sets = [v => BoxSet(P, settype()) for v in vertices(digraph)]
    for (key,val) in tiles.vals
        push!(last(sets[val]), key)
    end
    
    return MetaGraph(
        digraph,
        sets,
        edge_data,
        P
    )
end

function MetaGraph(adj::AbstractMatrix, tiles::BoxFun)
    MetaGraph(DiGraph(adj), tiles)
end

end # module
