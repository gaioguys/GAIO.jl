function relative_attractor(g::BoxMap, boxset::BoxSet, depth::Int)
    for k = 1:depth
        boxset = subdivide(boxset)
        boxset = g(boxset; target=boxset)
    end

    return boxset
end

function unstable_set!(g::BoxMap, boxset::BoxSet)
    boxset_new = boxset

    while !isempty(boxset_new)
        boxset_new = g(boxset_new)

        setdiff!(boxset_new, boxset)
        union!(boxset, boxset_new)
    end

    return boxset
end

function chain_recurrent_set(g::BoxMap, boxset::BoxSet, depth::Int)
    for k in 1:depth
        boxset = subdivide(boxset)
        G = transition_graph(g, boxset)
        boxset = strongly_connected_components(G)
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

function cover_roots(g, g_jacobian, boxset::BoxSet, points, depth::Int)
    for k in 1:depth
        boxset = subdivide(boxset)
        g_k = PointDiscretizedMap(x -> adaptive_newton_step(g, g_jacobian, x, k), points)
        boxset = g_k(boxset)
    end

    return boxset
end

"""
    relative_attractor(boxset::BoxSet, g; T, num_points=20, ϵ=1e-6) -> BoxFun

Compute the Finite Time Lyapunov Exponent for every box in boxset.

Arguments:
* `boxset` is the starting BoxSet.
* `g` is a point map (not a BoxMap) describing the dynamics.
* `T` is the length of the time interval under consideration.
* `num_points` is the number of points used in each box
* `ϵ` is the maximum size of a perturbation in each coordinate
"""
function finite_time_lyapunov_exponents(boxset::BoxSet, g; T, num_points=20, ϵ=1e-6)
    # TODO with some refactoring, this could maybe also use (some form of)
    # ParallelBoxIterator to implement parallelisation
    d = size(boxset.partition.domain.center)[1]
    dict = Dict{keytype(typeof(boxset.partition)),Float64}()
    sizehint!(dict,length(boxset))

    for box in boxset
        g_center = g(box.center)

        perturbations = ϵ * (2 * rand(Float64,(d,num_points)) .- 1)
        sample_points = box.center .+ perturbations

        distances = [norm(perturbations[:,i]) for i in 1:num_points]
        g_distances = [norm(g_center .- g(sample_points[:,i])) for i in 1:num_points]
        maximal_ftle = log(maximum(g_distances ./ distances)) / abs(T)
        # calculating the minimal ftle would also be straightforward (TODO?)
        dict[point_to_key(boxset.partition,box.center)] = maximal_ftle  # pretty sure this could be more efficient/idiomatic
    end
    
    return BoxFun(boxset.partition,dict)
end
