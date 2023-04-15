# Coherent Sets

### Mathematical Background

The following approach is decribed in [1]. 

Recalling the section on [Almost Invariant (metastable) Sets](@ref), we wish to understand sets which resist mixing over time. In the constext of autonomous dynamics, these are _almost invariant_ sets. However, in the context of nonautonomous dynamics these are often referred to as _coherent_ sets. To distinguish the maps for different times `t` we will write ``f_t`` to denote the dynamics at time ``t``. 

If the sets ``A_t`` and ``A_{t+\tau}`` satisfy ``A_t \approx f_{t+\tau}^{-1} (A_{t+\tau})`` then the set of points currently in ``A_t`` will be transported to ``A_{t+\tau}`` with little mixing, i.e. 
```math
\frac{m( A_t \cap f_{t+\tau}^{-1} (A_{t+\tau}) )}{m(A_t)} \approx 1 .
```
As in the previous sections, we will summarize this as an eigenproblem. We would like to translate this problem to one of the form 
```math
(f_t)_{\#}\,\mu_t \approx \mu_{t+\tau}
``` 
for some measures ``\mu_t,\ \mu_{t+\tau}`` with supports on ``A_t,\ A_{t+\tau}``, respectively. However, in general we cannot expect ``A_t = A_{t+\tau}`` and hence cannot expect ``\mu_t = \mu_{t+\tau}`` either. Therefore, the above equation is not an eigenproblem. 

Instead, our heuristic will be to push forward ``\mu_t`` using ``(f_{t})_{\#}`` to obtain something close to ``\mu_{t+\tau}``, and then pull back with the adjoint operator ``(f_{t})_{\#}^{\,*}`` to return something close to ``\mu_t``. (Analogously we could pull back ``\mu_{t+\tau}`` with ``(f_{t})_{\#}^{\,*}`` and then push forward with ``(f_{t})_{\#}``.) 

This leaves a new eigenproblem of the form ``(f_{t})_{\#}^{\,*} \, (f_{t})_{\#}`` (or ``(f_{t})_{\#} \, (f_{t})_{\#}^{\,*}``). These eigenvalues are precisely the right (or left) singular vectors of the operator ``(f_t)_{\#}``. 

### Example

We will consider the _periodically driven double-gyre map_

```@setup 1
using StaticArrays # hide
test_points = SVector{2,Float64}[               # hide
 (0.26421612450244525, -0.25171763065858643),   # hide
 (0.4570855415305548, -0.9944470807059835),     # hide
 (-0.7306293386393881, 0.06379824546718038),    # hide
 (-0.85365817202697, 0.003511957106970387),     # hide
 (0.8138787776391052, -0.7664619370413657),     # hide
 (-0.2656048895026608, 0.7623267304924846), # hide 
 (0.3437809335388058, -0.04027514156212453),    # hide
 (0.8999366397342172, -0.9475337796543073),     # hide
 (-0.30562250022109194, 0.6385081180020975),    # hide
 (0.5856626450795162, 0.934459449090036),       # hide
 (-0.0570952388870527, -0.6124402435996972),    # hide
 (0.8835618643488619, 0.33877549491860126),     # hide
 (0.7842181008345479, -0.2865702451606791),     # hide
 (0.45789504382722646, -0.1981801086140511),    # hide
 (-0.3709621343679155, 0.6094401439758141),     # hide
 (-0.7824441817855012, -0.0038770520790678553), # hide
 (0.10746570024408109, -0.022132632765053062),  # hide
 (-0.01683850957330124, 0.8655654869678553),    # hide
 (-0.08440158133026743, -0.17554973990426515),  # hide
 (-0.9262043546704484, 0.5106870713714742),     # hide
 (-0.6038030997879464, 0.41831615567516445),    # hide
 (0.16940903178018019, -0.626636883009092),     # hide
 (0.520026360934053, 0.3865846340173611),       # hide
 (-0.5823409268978248, -0.5940812669648463),    # hide
 (-0.12895805044268127, -0.766415470911298),    # hide
 (-0.858084556900655, 0.7777874203199997),      # hide
 (-0.37851170163453807, -0.704391110155435),    # hide
 (-0.44135552456739613, -0.3992132574381311),   # hide
 (0.22286176973262561, 0.48927750394661396),    # hide
 (-0.4399148899674712, 0.3714369719228312),     # hide
 (-0.7224409142472934, 0.9945315869571947),     # hide
 (0.49288810186172594, -0.8347990196625026)     # hide
] # hide

```

```@example 1
using GAIO

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

```@example 1
t₀, τ, t₁ = 0, 0.1, 1
Φₜ₀ᵗ¹(z) = Φ(z, t₀, τ, t₁)

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxPartition(domain, (256, 128))
S = cover(P, :)

F = BoxMap(:montecarlo, Φₜ₀ᵗ¹, domain)
F = BoxMap(:pointdiscretized, Φₜ₀ᵗ¹, domain, test_points) # hide

T = TransferOperator(F, S, S)

# we give Arpack some help converging to the singular values,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/2), 1000, ones(size(T, 2))
U, σ, V = svds(T; nsv=32, maxiter=maxiter, tol=tol, v0=v0)

σ
```

```@example 1
μ = U[2]
```

```@example 1
using Plots

p = plot(μ);

savefig(p, "sv1.svg"); nothing # hide
```

![First left singular measure](sv1.svg)

```@example 1
A1 = BoxSet(P, Set(key for key in keys(μ) if μ[key] ≥ 0))
A2 = BoxSet(P, Set(key for key in keys(μ) if μ[key] < 0))

p = plot(A1, color=:red);
p = plot!(A2, color=:blue, linecolor=:blue);

savefig(p, "coherent.svg"); nothing # hide
```

![Coherent Sets](coherent.svg)


### References

[1] Froyland, G., Padberg-Gehle, K. (2014). Almost-Invariant and Finite-Time Coherent Sets: Directionality, Duration, and Diffusion. In: Bahsoun, W., Bose, C., Froyland, G. (eds) Ergodic Theory, Open Dynamics, and Coherent Structures. Springer Proceedings in Mathematics & Statistics, vol 70. Springer, New York, NY. https://doi.org/10.1007/978-1-4939-0419-8_9
