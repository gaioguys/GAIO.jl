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
    expon(h, k=1, ϵ=0.2, δ=0.1)

Return a rough estimate of how many Newton steps 
should be taken, given a step size h. 
"""
function expon(h, k=1, ϵ=0.2, δ=0.1)
    n = log( ϵ * (1/2)^k ) / log( maximum((1 - h, δ)) )
    return Int(ceil(n))
end

"""
    adaptive_newton_step(g, g_jacobian, x, k=1)

Return one step of the adaptive Newton algorithm for the point `x`. 
"""
@muladd function adaptive_newton_step(g, g_jacobian, x, k=1)
    Dg = g_jacobian(x)
    d = Dg \ g(x)
    h = armijo_rule(g, g_jacobian, x, d)
    n = expon(h, k)
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
        f(x) = adaptive_newton_step(g, Dg, x, k)
        F = BoxMap(f, domain)
        B = F(B)
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
for some function ``f : \mathbb{R}^N \to \mathbb{R}``. 
    
The starting BoxSet `B` should (coarsely) cover 
the manifold. 
"""
function cover_manifold(f, B₀::BoxSet{Box{N,T},Q,S}; steps=12) where {N,T,Q,S}
    B = copy(B₀)
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        P = B.partition
        @floop for key in keys(B)
            c, r = key_to_box(P, key)
            fint = f(c .± r)
            if has_zero(fint)
                @reduce( image = S() ⊔ key )
            end
        end
        B = BoxSet(P, image)
    end
    return B
end
