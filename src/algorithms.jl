"""
    relative_attractor(F::BoxMap, B::BoxSet; steps=12) -> BoxSet

Compute the attractor relative to `B`. `B` should be 
a (coarse) covering of the relative attractor, e.g. 
`B = cover(P, :)` for a partition `P`.
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
a small box surrounding a fixed point of `F`. The partition must 
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

Compute the chain recurrent set over the box set `B`. 
`B` should be a (coarse) covering of the relative attractor, 
e.g. `B = cover(P, :)` for a partition `P`.
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
    preimage(F::BoxMap, B::BoxSet, Q::BoxSet) -> BoxSet

Compute the (restricted to `Q`) preimage of `B` under `F`, i.e.
```math
F^{-1} (B) \cap Q . 
```
Note that the larger ``Q`` is, the more calculation time required. 
"""
function preimage(F::BoxMap, B::BoxSet, Q::BoxSet)
    μ = BoxFun(B)
    T = TransferOperator(F, Q, B)
    return BoxSet(T'μ)
end

Core.@doc raw"""
    preimage(F::BoxMap, B::BoxSet) -> BoxSet

Efficiently compute 
```math
F^{-1} (B) \cap B . 
``` 
Significantly faster than calling `preimage(F, B, B)`. 

!!! warning "This is not the entire preimage in the mathematical sense!"
    `preimage(F, B)` computes the RESTRICTED preimage
    ``F^{-1} (B) \cap B``, NOT the full preimage 
    ``F^{-1} (B)``. 
"""
function preimage(F::BoxMap, B::BoxSet)
    P = TransferOperator(F, B, B)
    G = Graph(P)
    C⁻ = vec( sum(P.mat, dims=2) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
    C = findall(C⁻)
    return BoxSet(G, C)
end

Core.@doc raw"""
    symmetric_image(F::BoxMap, B::BoxSet) -> BoxSet

Efficiently compute 
```math
F (B) \cap B \cap F^{-1} (B) . 
```
Internally performs the following computation 
(though more efficiently) 
```julia
# create a measure with support over B
μ = BoxFun(B)

# compute transfer weights (restricted to B)
T = TransferOperator(F, B, B)

C⁺ = BoxSet(T*μ)    # support of pushforward measure
C⁻ = BoxSet(T'μ)    # support of pullback measure

C = C⁺ ∩ C⁻
```
"""
function symmetric_image(F::BoxMap, B::BoxSet)
    P = TransferOperator(F, B, B)
    G = Graph(P)
    C⁺ = vec( sum(P.mat, dims=1) .> 0 ) # C⁺ = B ∩ F(B)
    C⁻ = vec( sum(P.mat, dims=2) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
    C = findall(C⁺ .& C⁻)   # C  =  C⁺ ∩ C⁻  =  F(B) ∩ B ∩ F⁻¹(B)
    return BoxSet(G, C)
end

"""
    maximal_forward_invariant_set(F::BoxMap, B::BoxSet; steps=12)

Compute the maximal forward invariant set contained in `B`. 
`B` should be a (coarse) covering of a forward invariant set, 
e.g. `B = cover(P, :)` for a partition `P`.
"""
function maximal_forward_invariant_set(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    F⁻¹(B) = preimage(F, B)
    B = copy(B₀)
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        B = F⁻¹(B)  # is technically B ∩ F⁻¹(B)
    end
    return B
end

"""
    maximal_invariant_set(F::BoxMap, B::BoxSet; steps=12)

Compute the maximal invariant set contained in `B`. 
`B` should be a (coarse) covering of an invariant set, 
e.g. `B = cover(P, :)` for a partition `P`.
"""
function maximal_invariant_set(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    B = copy(B₀)
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        B = symmetric_image(F, B)    # F(B) ∩ B ∩ F⁻¹(B)
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
        @floop for key in B.set
            box = key_to_box(P, key)
            int = IntervalBox(box)
            fint = f(int)
            if contains_zero(fint)
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

"""
    nth_iterate_jacobian(f, Df, x, n; return_QR=false) -> Z[, R]

Compute the Jacobian of the `n`-times iterated function 
`f ∘ f ∘ ... ∘ f` at `x` using a QR iteration based on [1]. 
Requires an approximation `Df` of the jacobian of `f`, e.g. 
`Df(x) = ForwardDiff.jacobian(f, x)`. 
Optionally, return the QR decomposition. 

[1] Dieci, L., Russell, R. D., Van Vleck, E. S.: "On the 
Computation of Lyapunov Exponents for Continuous Dynamical 
Systems," submitted to SIAM J. Numer. Ana. (1993).
"""
function nth_iterate_jacobian(f, Df, x, n; return_QR=false)
    N, T = length(x), eltype(x)
    fx = x

    Z = Matrix{T}(I(N))
    ZR = Matrix{T}(I(N))

    Q = Matrix{T}(I(N))
    R = Matrix{T}(I(N))

    for i in 0:n
        decomp = qr(Z)
        Q .= decomp.Q
        R .= decomp.R
        fixqr!(Q, R)
        Z = Df(fx) * Q
        ZR = ZR * R
        i < n && (fx = f(fx))
    end

    Z = Q * ZR
    return return_QR ? (Z, Q, R) : Z
end

"""
    fixqr!(Q, R)

Adjust a QR-decomposition such that the 
R-factor has positive diagonal entries. 
"""
function fixqr!(Q, R)
    d = diag(R)
    Q[:, d .< 0] .*= -1
    R[d .< 0, :] .*= -1
    return Q, R
end

Core.@doc raw"""
    finite_time_lyapunov_exponents(f, Df, μ::BoxFun; n=8) -> σ

Compute the Lyapunov exponents using a spatial integration 
method [1] based on Birkhoff's ergodic theorem. Computes 
```math
\sigma_j = \frac{1}{n} \int \log R_{jj}( Df^n (x) ) \, dμ (x), \quad j = 1, \ldots, d
```
with respect to an ergodic invariant measure ``\mu``. 

[1] Beyn, WJ., Lust, A. A hybrid method for computing 
Lyapunov exponents. Numer. Math. 113, 357–375 (2009). 
https://doi.org/10.1007/s00211-009-0236-4
"""
function finite_time_lyapunov_exponents(f, Df, μ::BoxFun{E}; n=8) where {N,T,E<:Box{N,T}}
    Dfⁿ(x) = nth_iterate_jacobian(f, Df, x, n; return_QR=true)
    a = sum(μ; init=zeros(N)) do x
        _, _, R = Dfⁿ(x)
        log.(diag(R))
    end
    sort!(a, rev=true)
    a ./= n
    return a
end

"""
    box_dimension(boxsets) -> D

For an iterator `boxsets` of (successively finer) 
`BoxSet`s, compute the box dimension `D`. 

#### Example
```julia
# F is some BoxMap, S is some BoxSet
box_dimension( relative_attractor(F, S, steps=k) for k in 1:20 )
```
"""
function box_dimension(boxsets)
    logϵ, box_dim = Float64[], Float64[]
    for boxset in boxsets
        ϵ = 2 * maximum(max_radius(boxset))
        logϵ_n = 1 / log(1/ϵ)
        N = length(boxset)
        push!(logϵ, logϵ_n)
        push!(box_dim, log(N)*logϵ_n)
    end
    logK, D = linreg(logϵ, box_dim)
    return D
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

Core.@doc raw"""
    seba(V::Vector{<:BoxFun}, Rinit=nothing; which=partition_disjoint, maxiter=5000) -> S, A

Construct a sparse eigenbasis approximation of `V`, as described in 
[1]. Returns an `Array` of `BoxFun`s corresponding to the eigenbasis, 
as well as a maximum-likelihood `BoxFun` that maps a box to the 
element of `S` which has the largest value over the support. 

The keyword `which` is used to set the threshholding heuristic, 
which is used to extract a partition of the supports from the 
sparse basis. Builtin options are
```julia
partition_unity, partition_disjoint, partition_likelihood
```
which are all exported functions. 
"""
function seba(V::AbstractArray{U}, Rinit=nothing; which=partition_disjoint) where {B,K,W,Q,D<:OrderedDict,U<:BoxFun{B,K,W,Q,D}}
    #supp = BoxSet(V[1])
    #all(x -> BoxSet(x) == supp, V) || throw(DomainError(V, "Supports of BoxFuns do not match."))
    supp = union((BoxSet(μ) for μ in V)...)

    V̄ = [μ[key] for key in supp.set, μ in V]
    S̄, R = seba(V̄, Rinit)
    S̄, Ā, τ = which(S̄)

    S = [BoxFun(supp, S̄[:, i]) for i in 1:size(S̄, 2)]
    A = BoxFun(supp, Ā)
    return S, A
end

function seba(V::AbstractArray{U}, Rinit=nothing; which=partition_unity) where {B,K,W,Q,D,U<:BoxFun{B,K,W,Q,D}}
    V = [
        BoxFun( V[i].partition, OrderedDict(V[i]) )
        for i in 1:length(V)
    ]   # convert to ordered collections to guarantee deterministic iteration order
    return seba(V, Rinit; which=which)
end

function partition_unity(S)
    S .= max.(S, 0)
    S_descend = sort(S, dims=2, rev=true)
    S_sum = cumsum(S_descend, dims=2)
    τᵖᵘ = maximum(S_descend[S_sum .> 1], init=zero(eltype(S)))
    S[S .≤ τᵖᵘ] .= 0
    A = argmax.(eachrow(S))
    return S, A, τᵖᵘ
end

function partition_disjoint(S)
    S .= max.(S, 0)
    S_descend = sort(S, dims=2, rev=true)
    τᵈᵖ = maximum(S_descend[:, 2], init=zero(eltype(S)))
    S[S .≤ τᵈᵖ] .= 0
    A = argmax.(eachrow(S))
    return S, A, τᵈᵖ
end

function partition_likelihood(S)
    A = argmax.(eachrow(S))
    M = S[:, A]
    A[M .≤ 0] .= 0
    S .= 0
    r = size(S, 2)
    for i in 1:r
        S[A .== i, i] .= M[A .== i]
    end
    return S, A, 0.
end

Core.@doc raw"""
    seba(V::Matrix{<:Real}, Rinit=nothing, maxiter=5000) -> S, R

Construct a sparse approximation of the basis `V`, as described in 
[1]. Returns matrices ``S``, ``R`` such that
```math
\frac{1}{2} \| V - SR \|_F^2 + \mu \| S \|_{1,1}
```
is minimized,
where ``\mu \in \mathbb{R}``, ``\| \cdot \|_F`` is the Frobenuius-norm, 
and ``\| \cdot \|_{1,1}`` is the element sum norm, and ``R`` 
is orthogonal. See [1] for further information on the argument 
`Rinit`, as well as a description of the algorithm. 

[1] Gary Froyland, Christopher P. Rock, and Konstantinos Sakellariou. 
Sparse eigenbasis approximation: multiple feature extraction 
across spatiotemporal scales with application to coherent set 
identification. Communications in Nonlinear Science and Numerical 
Simulation, 77:81-107, 2019. https://arxiv.org/abs/1812.02787
"""
function seba(V::AbstractArray{U}, Rinit=nothing, maxiter=5000) where {U}
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

"""
    linreg(xs, ys)

Simple one-dimensional lunear regression used to 
approximate box dimension. 
"""
function linreg(xs, ys)
    n = length(xs)
    n == length(ys) || throw(DimensionMismatch())

    sum_x, sum_y = sum(xs), sum(ys)
    sum_xy, sum_x2 = xs'ys, xs'xs

    m = ( n*sum_xy - sum_x*sum_y ) / ( n*sum_x2 - sum_x^2 )
    b = ( sum_x*sum_y - m*sum_x^2 ) / ( n*sum_x )

    return m, b
end
