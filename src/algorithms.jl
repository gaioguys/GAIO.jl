"""
    relative_attractor(F::BoxMap, B::BoxSet; steps=12) -> BoxSet

Compute the attractor relative to `B`. Generally, `B` should be 
a box set containing the whole partition `P`, ie `B = P[:]`.
"""
function relative_attractor(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    B = copy(B₀)
    for k = 1:steps
        B = subdivide(B, (k % N) + 1)
        B = B ∩ F(B)
    end
    return B
end

"""
    unstable_set(F::BoxMap, B::BoxSet) -> BoxSet

Compute the unstable set for a box set `B`. Generally, `B` should be 
a small box surrounding a fixed point of `F`. The partition should 
be fine enough, since no subdivision occurs in this algorithm. 
"""
function unstable_set(F::BoxMap, B::BoxSet)
    B₀ = copy(B)
    B₁ = copy(B)
    while !isempty(B₁)
        B₁ = F(B₁)
        setdiff!(B₁, B₀)
        union!(B₀, B₁)
    end
    return B₀
end

"""
    chain_recurrent_set(F::BoxMap, B::BoxSet; steps=12) -> BoxSet

Compute the chain recurrent set over the box set `B`. Generally, 
`B` should be a box set containing the whole partition `P`, 
ie `B = P[:]`. 
"""
function chain_recurrent_set(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    B = copy(B₀)
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        P = TransferOperator(F, B, B)
        G = Graph(P)
        B = union_strongly_connected_components(G)
    end
    return B
end

Core.@doc raw"""
    armijo_rule(g, Dg, x, d, σ=1e-4, ρ=0.8, α₀=0.05, α₁=1.0)

Find a step size multiplier ``\alpha \in (\alpha_0, \alpha_1]`` 
such that 
```math
g(x + \alpha d) - g(x) \leq \alpha \sigma \, Dg(x) \cdot d
```
This is done by initializing ``\alpha = 1`` and testing the 
above condition. If it is not satisfied, scale ``\alpha`` 
by some constant ``\rho < 1`` (i.e. set 
``\alpha = \rho \cdot \alpha``), and test the condition 
again. 
"""
@muladd function armijo_rule(g, Dg, x, d, σ=1e-4, ρ=0.8, α₀=0.05, α₁=1.0)
    gx, Dgx = g(x), Dg(x)
    α = α₁
    while any(g(x + α * d) .> gx + σ * α * Dgx' * d) && α > α₀
        α = ρ * α
    end
    return α
end

"""
    expon(h, ϵ=0.2, σ=1.0, δ=0.1)

Return a rough estimate of how many Newton steps 
should be taken, given a step size h. 
"""
function expon(h, ϵ=0.2, σ=1.0, δ=0.1)
    n = log( ϵ * (1/2)^σ ) / log( maximum((1 - h, δ)) )
    return Int(ceil(n))
end

"""
    adaptive_newton_step(g, g_jacobian, x, k=1)

Return one step of the adaptive Newton algorithm for the point `x`. 
"""
@muladd function adaptive_newton_step(g, g_jacobian, x)
    Dg = g_jacobian(x)
    d = Dg \ g(x)
    h = armijo_rule(g, g_jacobian, x, d)
    n = expon(h)
    for _ in 1:n
        Dg = g_jacobian(x)
        x = x - h * (Dg \ g(x))
    end
    return x
end

"""
    cover_roots(g, Dg, B::BoxSet; steps=12) -> BoxSet

Compute a covering of the roots of `g` within the 
partition `P`. Generally, `B` should be 
a box set containing the whole partition `P`, ie 
`B = cover(P, :)`, and should contain a root of `g`. 
"""
function cover_roots(g, Dg, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    B = copy(B₀)
    domain = B.partition.domain
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        f = x -> adaptive_newton_step(g, Dg, x, k)
        F_k = BoxMap(f, domain, no_of_points = 40)
        B = F_k(B)
    end
    return B
end

Core.@doc raw"""
    cover_manifold(f, B::BoxSet; steps=12)

Use interval arithmetic to compute a covering of 
an implicitly defined manifold ``M`` of the form 
```math
f(M) \equiv 0
```
for some function ``f : \mathbb{R}^N \rigtharrow \mathbb{R}``. 
    
The starting BoxSet `B` should (coarsely) cover 
the manifold. 
"""
function cover_manifold(f, B₀::BoxSet{Box{N,T},Q,S}; steps=12) where {N,T,Q,S}
    B = copy(B₀)
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        P = B.partition
        @floop for key in B.set
            c, r = key_to_box(P, key)
            box = IntervalBox(c .± r ...)
            fbox = f(box)
            if sign(fbox.lo * fbox.hi) ≤ 0
                @reduce( image = S() ⊔ key )
            end
        end
        B = BoxSet(P, image)
    end
    return B
end

"""
    finite_time_lyapunov_exponents(F::SampledBoxMap, boxset::BoxSet) -> BoxFun

Compute the Finite Time Lyapunov Exponent for 
every box in `boxset`, where `F` represents a time-`T` 
integration of some continuous dynamical system. 
It is assumed that all boxes in `boxset` have radii 
of some fixed order ϵ. 
"""
function finite_time_lyapunov_exponents(F::SampledBoxMap, B::BoxSet{R,Q,S}; T) where {N,V,R<:Box{N,V},Q,S}
    P, D = B.partition, Dict{keytype(Q),Float64}
    @floop for key in B.set
        c, r = key_to_box(P, key)
        fc = F.map(c)
        ftle = -Inf
        for p in F.domain_points(c, r)
            ϵ = norm(c .- p)
            ϵ == 0 && continue
            fp = F.map(p)
            ftle_pot = log( norm(fc .- fp) / ϵ ) / abs(T)
            ftle = max(ftle, ftle_pot)
        end
        @reduce( vals = D() ⊔ (key => ftle) )
    end
    return BoxFun(B.partition, vals)
end

# Runge-Kutta scheme of 4th order
const half, sixth, third = Float32.((1/2, 1/6, 1/3))

"""
    rk4(f, x, τ)

Compute one step with step size `τ` of the classic 
fourth order Runge-Kutta method. 
"""
@muladd @inline function rk4(f, x, τ)
    τ½ = τ * half

    k = f(x)
    dx = @. k * sixth

    k = f(@. x + τ½ * k)
    dx = @. dx + k * third

    k = f(@. x + τ½ * k)
    dx = @. dx + k * third

    k = f(@. x + τ * k)
    dx = @. dx + k * sixth

    return @. x + τ * dx
end

"""
    rk4_flow_map(f, x, step_size=0.01, steps=20)

Perform `steps` steps of the classic Runge-Kutta fourth order method,
with step size `step_size`. 
"""
@inline function rk4_flow_map(f, x, step_size=0.01f0, steps=20)
    for _ in 1:steps
        x = rk4(f, x, step_size)
    end
    return x
end

function SEBA(V::AbstractArray{<:BoxFun}, Rinit=nothing)
    P = V[1].partition
    all(μ -> μ.partition == P, V) || throw(DomainError(V, "Partitions of BoxFuns do not match. "))
    supp = union((keys(μ) for μ in V)...)

    V̄ = [μ[key] for key in supp, μ in V]
    S̄, R = SEBA(V̄, Rinit)

    S̄ .= max.(S̄, 0)
    S_descend = sort(S̄, dims=1, rev=true)
    τdp = maximum(S_descend[:, 2])
    S̄[S̄ .< τdp] .= 0

    S = [BoxFun(V[i], S̄[:, i]) for i in 1:size(S̄, 2)]
    return S
end

function SEBA(V::AbstractMatrix, Rinit=nothing)

    # Inputs: 
    # V is pxr matrix (r vectors of length p as columns)
    # Rinit is an (optional) initial rotation matrix.

    # Outputs:
    # S is pxr matrix with columns approximately spanning the column space of V
    # R is the optimal rotation that acts on V, which followed by thresholding, produces S

    maxiter = 5000   #maximum number of iterations allowed
    F = qr(V) # Enforce orthonormality
    V = Matrix(F.Q)
    p, r = size(V)
    μ = 0.99 / sqrt(p)

    S = zeros(size(V))
    # Perturb near-constant vectors
    for j = 1:r
        if maximum(V[:,j]) - minimum(V[:,j]) < 1e-14
            V[:,j] = V[:,j] .+ (rand(p, 1) .- 1 / 2) * 1e-12
        end
    end

    # Initialise rotation
    if Rinit ≡ nothing
        Rnew = Matrix(I, r, r)
    else
        # Ensure orthonormality of Rinit
        F = svd(Rinit)
        Rnew = F.U * F.Vt
    end

    R = zeros(r, r)
    iter = 0
    while norm(Rnew - R) > 1e-14 && iter < maxiter
        iter = iter + 1
        R = Rnew
        Z = V * R'
        # Threshold to solve sparse approximation problem
        for i = 1:r
            Si = sign.(Z[:,i]) .* max.(abs.(Z[:,i]) .- μ, zeros(p))
            S[:,i] = Si / norm(Si)
        end
        # Polar decomposition to solve Procrustes problem
        F = svd(S' * V, full=false)
        Rnew = F.U * F.Vt
    end

    # Choose correct parity of vectors and scale so largest value is 1
    for i = 1:r
        S[:,i] = S[:,i] * sign(sum(S[:,i]))
        S[:,i] = S[:,i] / maximum(S[:,i])
    end

    # Sort so that most reliable vectors appear first
    ind = sortperm(vec(minimum(S, dims=1)), rev=true)
    S = S[:, ind]

    return S, R

end
