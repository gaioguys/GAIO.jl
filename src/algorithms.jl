function relative_attractor(boxset::BoxSet, g::BoxMap, depth::Int)
    for k = 1:depth
        boxset = subdivide(boxset)
        boxset = g(boxset; target=boxset)
    end

    return boxset
end

function unstable_set!(boxset::BoxSet, g::BoxMap)
    boxset_new = boxset

    while !isempty(boxset_new)
        boxset_new = g(boxset_new)

        setdiff!(boxset_new, boxset)
        union!(boxset, boxset_new)
    end

    return boxset
end

function strongly_connected_vertices(edges)
    connected_vertices = Int[]
    graph = SimpleDiGraph(Edge.(edges))
    scc = strongly_connected_components(graph)

    for k in 1:length(scc)
        n = length(scc[k])
        if n > 1
            for i in 1:n
                push!(connected_vertices, scc[k][i])
            end
        end
    end

    for k in vertices(graph)
        if (k,k) in edges
            push!(connected_vertices, k)
        end
    end

    return connected_vertices
end

function chain_recurrent_set(boxset::BoxSet, g::BoxMap, depth::Int)
    for k in 1:depth
        boxset = subdivide(boxset)
        edges, vertex_to_key = map_boxes_to_edges(g, boxset)
        connected_vertices = strongly_connected_vertices(edges)
        boxset = BoxSet(boxset.partition, Set(vertex_to_key[connected_vertices]))
    end

    return boxset
end
