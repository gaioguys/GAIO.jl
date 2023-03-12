# Almost Invariant and (Lagrangian) Coherent Sets

### Mathematical Background

(TODO)

### Example

We already constructed almost invariant sets in the section on [Sparse Eigenbasis Approximation (SEBA)](@ref). However, we will use the symmetrized transfer operator in this example. We will continue using the periodically driven double-gyre introduced in the section on [Finite Time Lyapunov Exponents](@ref). See that code block for the definition of the map. 

```@setup 1
using GAIO
using Plots

const A, ϵ, ω = 0.25, 0.25, 2π
function double_gyre(x, y, t)
    f(x, t)  =  ϵ * sin(ω*t) * x^2 + (1 - 2ϵ * sin(ω*t)) * x
    df(x, t) = 2ϵ * sin(ω*t) * x   + (1 - 2ϵ * sin(ω*t))

    return (
        -π * A * sin(π * f(x, t)) * cos(π * y),
         π * A * cos(π * f(x, t)) * sin(π * y) * df(x, t),
         1
    )
end

double_gyre((x, y, t)) = double_gyre(x, y, t)

# create a point map ℝ² → ℝ² that 
# integrates the vector field 
# with fixed start time t₀, step size τ 
# until a fixed end time t₁ is reached
function Φ((x₀, y₀), t₀, τ, t₁)
    z = (x₀, y₀, t₀)
    for _ in t₀ : τ : t₁-τ
        z = rk4(double_gyre, z, τ)
    end
    (x₁, y₁, t₁) = z
    return (x₁, y₁)
end
```

```julia
t₀, τ, t₁ = 0, 0.1, 2
Φₜ₀ᵗ¹(z) = Φ(z, t₀, τ, t₁)

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxPartition(domain, (256, 128))
S = cover(P, :)
F = BoxMap(:montecarlo, Φₜ₀ᵗ¹, domain, no_of_points=128)
T = SymmetricTransferOperator(F, S, S)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(T, 2))
λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)
μ = log ∘ abs ∘ ev[2]

p = plot(μ)

savefig("second_eigvec.svg"); nothing # hide
```

(insert plot here)

We notice there are two "blobs" defining the second eigenmeasure. These correspond to the almost invariant sets; there are two "vortices" where mass flows in a circular pattern and doesn't mix with the rest of the domain. We wish to isolate these blobs

```julia
ev_seba = seba(ev, which=partition_unity)

p1 = plot(ev_seba[1])
p2 = plot(ev_seba[2])
p = plot(p1, p2, layout=2, size=(1200,600))

savefig("seba.svg"); nothing # hide
```

(insert plot here)

### References

[1] Froyland, Gary & Padberg-Gehle, Kathrin. (2014). Almost-Invariant and Finite-Time Coherent Sets: Directionality, Duration, and Diffusion. 10.1007/978-1-4939-0419-8_9. 
