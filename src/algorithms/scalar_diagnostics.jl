"""
    finite_time_lyapunov_exponents(F::SampledBoxMap, boxset::BoxSet) -> BoxMeasure

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
    return BoxMeasure(B.partition, vals)
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
    finite_time_lyapunov_exponents(f, Df, μ::BoxMeasure; n=8) -> σ

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
function finite_time_lyapunov_exponents(f, Df, μ::BoxMeasure{E}; n=8) where {N,T,E<:Box{N,T}}
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
