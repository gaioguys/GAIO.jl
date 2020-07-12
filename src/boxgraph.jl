struct BoxGraph{P <: BoxPartition,K}
    partition::P
    edges::Dict{Tuple{K,K},Rational{Int}}
end

function vertex_set(boxgraph::BoxGraph{P,K}) where {P,K}
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

    scc = LightGraphs.strongly_connected_components(graph)

    connected_vertices = Int[]

    for k in 1:length(scc)
        n = length(scc[k])
        if n > 1
            for i in 1:n
                push!(connected_vertices, scc[k][i])
            end
        elseif (scc[k][1], scc[k][1]) in keys(boxgraph.edges)
            push!(connected_vertices, scc[k][1])
        end
    end

    return BoxSet(boxgraph.partition, Set(int_to_key[connected_vertices]))
end

function matrix(boxgraph::BoxGraph)
    int_to_key = collect(vertex_set(boxgraph))
    key_to_int = invert_vector(int_to_key)

    n = length(int_to_key)
    mat = zeros(Rational{Int}, n, n)

    for (edge, weight) in boxgraph.edges
        mat[key_to_int[edge[1]], key_to_int[edge[2]]] = weight
    end

    ints_to_boxset = let part = boxgraph.partition, int_to_key = int_to_key
        ints -> BoxSet(part, Set(int_to_key[ints]))
    end

    return mat, ints_to_boxset
end
