struct BoxGraph{L<:BoxList,W}
    vertices::L
    edges::Dict{Tuple{Int,Int},W}
end

function strongly_connected_components(boxgraph::BoxGraph)
    graph = SimpleDiGraph(
        [Edge(edge[1], edge[2]) for edge in keys(boxgraph.edges)]
    )

    sccs = LightGraphs.strongly_connected_components(graph)

    connected_vertices = Int[]

    for scc in sccs
        if length(scc) > 1 || has_edge(graph, scc[1], scc[1])
            append!(connected_vertices, scc)
        end
    end

    return BoxSet(boxgraph.vertices[connected_vertices])
end

function matrix(boxgraph::BoxGraph{L,W}) where {L,W}
    num_edges = length(boxgraph.edges)

    I = Vector{Int}()
    sizehint!(I, num_edges)

    J = Vector{Int}()
    sizehint!(J, num_edges)

    V = Vector{W}()
    sizehint!(V, num_edges)

    for (edge, weight) in boxgraph.edges
        push!(I, edge[1])
        push!(J, edge[2])
        push!(V, weight)
    end

    n = length(boxgraph.vertices)
    return sparse(I, J, V, n, n)
end
