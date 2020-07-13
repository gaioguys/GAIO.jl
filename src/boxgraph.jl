struct BoxGraph{P <: BoxPartition,K,W}
    partition::P
    edges::Dict{Tuple{K,K},W}
end

function vertex_set(boxgraph::BoxGraph{P,K,W}) where {P,K,W}
    edges = boxgraph.edges

    result = Set{K}()
    sizehint!(result, 2*length(edges))

    for edge in keys(edges)
        push!(result, edge[1])
        push!(result, edge[2])
    end

    return result
end

function invert_vector(x::Vector{T}) where T
    dict = Dict{T,Int}()
    sizehint!(dict, length(x))

    for i in 1:length(x)
        dict[x[i]] = i
    end

    return dict
end

function strongly_connected_components(boxgraph::BoxGraph)
    int_to_key = collect(vertex_set(boxgraph))
    key_to_int = invert_vector(int_to_key)

    graph = SimpleDiGraph(
        [Edge(key_to_int[edge[1]], key_to_int[edge[2]]) for edge in keys(boxgraph.edges)]
    )

    sccs = LightGraphs.strongly_connected_components(graph)

    connected_vertices = Int[]

    for scc in sccs
        if length(scc) > 1 || has_edge(graph, scc[1], scc[1])
            append!(connected_vertices, scc)
        end
    end

    return BoxSet(boxgraph.partition, Set(int_to_key[connected_vertices]))
end

function matrix(boxgraph::BoxGraph{P,K,W}) where {P,K,W}
    int_to_key = collect(vertex_set(boxgraph))
    key_to_int = invert_vector(int_to_key)

    num_edges = length(boxgraph.edges)

    I = Vector{Int}()
    sizehint!(I, num_edges)

    J = Vector{Int}()
    sizehint!(J, num_edges)

    V = Vector{W}()
    sizehint!(V, num_edges)

    for (edge, weight) in boxgraph.edges
        push!(I, key_to_int[edge[1]])
        push!(J, key_to_int[edge[2]])
        push!(V, weight)
    end

    n = length(int_to_key)
    mat = sparse(I, J, V, n, n)

    ints_to_boxset = let part = boxgraph.partition, int_to_key = int_to_key
        ints -> BoxSet(part, Set(int_to_key[ints]))
    end

    return mat, ints_to_boxset
end
