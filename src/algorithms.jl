function relative_attractor(F::BoxMap, B::BoxSet{<:AbstractBoxPartition{Box{N,T}}}; steps=12) where {N,T}
    for k = 1:steps
        B = subdivide(B, (k%N)+1)
        B = B ∩ F(B)
    end
    return B
end

function unstable_set!(F::BoxMap, B::BoxSet)
    B_new = B
    while !isempty(B_new)
        B_new = F(B_new)
        setdiff!(B_new, B)
        union!(B, B_new)
    end
    return B
end

function chain_recurrent_set(F::BoxMap, B::BoxSet{<:AbstractBoxPartition{Box{N,T}}}; steps=12) where {N,T}
    for k in 1:steps
        B = subdivide(B, (k%N)+1)
        P = TransferOperator(F, B)
        B = strongly_connected_components(P)
    end
    return B
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

function cover_roots(g, Dg, B::BoxSet{<:AbstractBoxPartition{Box{N,T}}}; steps::Int=12) where {N,T}
    domain = B.partition.domain
    for k in 1:steps
        B = subdivide(B, (k%N)+1)
        f = x -> adaptive_newton_step(g, Dg, x, k)
        F_k = BoxMap(f, domain, no_of_points = 40)
        B = F_k(B)
    end
    return B
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

# Runge-Kutta scheme of 4th order
@muladd @inline function rk4(f, x, τ)
    τ½ = τ/2

    k = f(x)
    dx = @. k/6

    k = f(@. x+τ½*k)
    dx = @. dx + k/3

    k = f(@. x+τ½*k)
    dx = @. dx + k/3

    k = f(@. x+τ*k)
    dx = @. dx + k/6

    return @. x+τ*dx
end

@muladd @inline function rk4!(f, x, dx, k, τ, τp2)

       k .= f(x)
    @. dx = k * sixth
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τ * k

       k .= f(k)
    @. dx = dx + k * sixth
    @. x  = x + τ * dx

    return nothing
end

@inline function rk4_flow_map(f, u::SVector{N,T}; τ=0.01, steps=20) where {N,T}
    for _ in 1:steps
        u = rk4(f, u, τ)
    end
    return u
end

@inline function rk4_flow_map(f, u; τ=0.01, steps=20)
    du, k = similar(u), similar(u)
    τp2 = τ / 2
    for _ in 1:steps
        rk4!(f, u, du, k, τ, τp2)
    end
    return u
end

function rk4_flow_map(f, N::Int; τ=0.01, steps=20)
    unrolled_f = @generate_unrolled f N
    rk4_f(u) = rk4_flow_map(unrolled_f, u; τ=τ, steps=steps)
    simd_f = @generate_simd rk4_f N
    return simd_f
end
