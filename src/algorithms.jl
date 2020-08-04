"""
    relative_attractor(boxset::BoxSet, g::BoxMap, depth::Int) -> BoxSet

Compute an approximation of the relative global attractor of the set `boxset` w.r.t. the map `g`
using the subdivision algorithm (Algorithm 1 in [^DFJ2001]) for `depth` rounds.

See the [Algorithms and Mathematical Background](@ref) page of the documentation for more information.

[^DFJ2001]: Dellnitz, M.; Froyland, G.; Junge, O.: The algorithms behind GAIO - Set oriented numerical methods for dynamical systems,
B. Fiedler (ed.): Ergodic theory, analysis, and efficient simulation of dynamical systems, Springer (2001)
"""
function relative_attractor(boxset::BoxSet, g::BoxMap, depth::Int)
    for k = 1:depth
        boxset = subdivide(boxset)
        boxset = g(boxset; target=boxset)
    end

    return boxset
end

"""
    unstable_set!(boxset::BoxSet, g::BoxMap) -> BoxSet

Compute an approximation to the unstable manifold
using the continuation algorithm applied to `g` and `boxset` (Algorithm 3, step 2 in [^DFJ2001])
under the assumption that [`relative_attractor`](@ref) has been applied to `boxset` before (i.e. Algorithm 3, step 1 in [^DFJ2001] is done).
The `boxset` attribute is modified during the process.

See the [Algorithms and Mathematical Background](@ref) page of the documentation for more information.

[^DFJ2001]: Dellnitz, M.; Froyland, G.; Junge, O.: The algorithms behind GAIO - Set oriented numerical methods for dynamical systems,
B. Fiedler (ed.): Ergodic theory, analysis, and efficient simulation of dynamical systems, Springer (2001)
"""
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

"""
    chain_recurrent_set(boxset::BoxSet, g::BoxMap, depth::Int) -> BoxSet

Apply the subdivision algorithm for computing chain-recurrent sets (Algorithm 2 in [^DFJ2001])
for `depth` rounds, starting with the set `boxset` and using the BoxMap `g`.

See the [Algorithms and Mathematical Background](@ref) page of the documentation for more information.

[^DFJ2001]: Dellnitz, M.; Froyland, G.; Junge, O.: The algorithms behind GAIO - Set oriented numerical methods for dynamical systems,
B. Fiedler (ed.): Ergodic theory, analysis, and efficient simulation of dynamical systems, Springer (2001)
"""
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

"""
    cover_roots(boxset::BoxSet, g, g_jacobian, points, depth::Int) -> BoxSet

Apply `depth` rounds of the root covering algorithm [^DSS2002] to find zeros
of the given function `g` with Jacobian `g_jacobian`. This algorithm combines
the subdivision idea with a Newton method.

Arguments:
* `boxset` is the starting BoxSet (normally 0-depth).
* `g` is a point map (not a BoxMap). (used for the Newton step)
* `g_jacobian` is the Jacobian matrix of `g`. (used for the Newton step)
* `points` is the points used for the PointDiscretizedMap which wraps the Newton step function.
* `depth` is the number of rounds we apply the algorithm. If `boxset` has depth 0, then `depth` equals the depth of the returned BoxSet.

See the [Algorithms and Mathematical Background](@ref) page of the documentation for more information.

[^DSS2002]: Dellnitz, Michael, Oliver Schütze, and Stefan Sertl. "Finding zeros by multilevel subdivision techniques." IMA Journal of Numerical Analysis 22.2, pp. 167-185 (2002)
"""
function cover_roots(boxset::BoxSet, g, g_jacobian, points, depth::Int)
    for k in 1:depth
        boxset = subdivide(boxset)
        g_k = PointDiscretizedMap(x -> adaptive_newton_step(g, g_jacobian, x, k), points)
        boxset = g_k(boxset)
    end

    return boxset
end
