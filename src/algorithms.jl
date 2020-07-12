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

function adaptive_newton_step(g, g_jacobian, x, k)
    armijo_rule = (g, x, α, σ, ρ) -> begin
        Dg = g_jacobian(x)
        d = Dg\g(x)
        while any(g(x + α * d) .> g(x) + σ * α * Dg' * d) && α > 0.1
            α = ρ * α
        end
        
        return α
    end
    h = armijo_rule(g, x, 1.0, 1e-4, 0.8)

    expon = (ϵ, σ, h, δ) -> Int(ceil(log(ϵ * (1/2)^σ)/log(maximum((1 - h, δ)))))
    n = expon(0.2, k, h, 0.1)

    for _ in 1:n
        Dg = g_jacobian(x)
        x = x - h * (Dg \ g(x))
    end
    
    return x
end

function cover_roots(boxset::BoxSet, g, g_jacobian, points, depth::Int)
    for k in 1:depth
        boxset = subdivide(boxset)
        g_k = PointDiscretizedMap(x -> adaptive_newton_step(g, g_jacobian, x, k), points)
        boxset = g_k(boxset)
    end

    return boxset
end
