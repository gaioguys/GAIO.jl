# Coherent Sets

### Mathematical Background

The following approach is decribed in [1]. 

Recalling the section on [Almost Invariant (metastable) Sets](@ref), we wish to understand sets which resist mixing over time. In the constext of autonomous dynamics, these are _almost invariant_ sets. However, in the context of nonautonomous dynamics these are often referred to as _coherent_ sets. To distinguish the maps for different times `t` we will write ``f_t`` to denote the dynamics at time ``t``. 

If the sets ``A_t`` and ``A_{t+\tau}`` satisfy ``A_t \approx f_{t}^{-1} (A_{t+\tau})`` then the set of points currently in ``A_t`` will be transported to ``A_{t+\tau}`` with little mixing, i.e. 
```math
\frac{m( A_t \cap f_{t}^{-1} (A_{t+\tau}) )}{m(A_t)} \approx 1 .
```
As in the previous sections, we will summarize this as an eigenproblem. We would like to translate this problem to one of the form 
```math
(f_t)_{\#}\,\mu_t \approx \mu_{t+\tau}
``` 
for some measures ``\mu_t,\ \mu_{t+\tau}`` with supports on ``A_t,\ A_{t+\tau}``, respectively. However, in general we cannot expect ``A_t = A_{t+\tau}`` and hence cannot expect ``\mu_t = \mu_{t+\tau}`` either. Therefore, the above equation is not an eigenproblem. 

Instead, our heuristic will be to push forward ``\mu_t`` using ``(f_{t})_{\#}`` to obtain something close to ``\mu_{t+\tau}``, and then pull back with the adjoint operator ``(f_{t})_{\#}^{\,*}`` to return something close to ``\mu_t``. (Analogously we could pull back ``\mu_{t+\tau}`` with ``(f_{t})_{\#}^{\,*}`` and then push forward with ``(f_{t})_{\#}``.) 

This leaves a new eigenproblem of the form ``(f_{t})_{\#}^{\,*} \, (f_{t})_{\#}`` (or ``(f_{t})_{\#} \, (f_{t})_{\#}^{\,*}``). These eigenvalues are precisely the right (or left) singular vectors of the operator ``(f_t)_{\#}``. 

### Example

We will consider the _periodically driven double-gyre_ described in the section on [Almost Invariant (metastable) Sets](@ref). See that page for details on the map. 

```@setup 1
using GAIO, LinearAlgebra

const A, ϵ, ω = 0.25, 0.25, 2π

f(x, t)  =  ϵ * sin(ω*t) * x^2 + (1 - 2ϵ * sin(ω*t)) * x
df(x, t) = 2ϵ * sin(ω*t) * x   + (1 - 2ϵ * sin(ω*t))

double_gyre(x, y, t) = (
    -π * A * sin(π * f(x, t)) * cos(π * y),
     π * A * cos(π * f(x, t)) * sin(π * y) * df(x, t)
)

# autonomize the ODE by adding a dimension
double_gyre((x, y, t)) = (double_gyre(x, y, t)..., 1)

# nonautonomous flow map: reduce back to 2 dims
function Φ((x, y), t, τ, steps)
    (x, y, t) = rk4_flow_map(double_gyre, (x, y, t), τ, steps)
    return (x, y)
end
```

```@example 1
t₀, τ, steps = 0, 0.1, 20
t₁ = t₀ + τ * steps
Tspan = t₁ - t₀
Φₜ₀ᵗ¹(z) = Φ(z, t₀, τ, steps)

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxGrid(domain, (256, 128))
S = cover(P, :)

F = BoxMap(:grid, Φₜ₀ᵗ¹, domain, n_points=(6,6))

T = TransferOperator(F, S, S)

# we need to rescale the operator so that we
# can be certain that the second-largest 
# singular value is the desired one
function rescale!(T::TransferOperator)
    M = T.mat
    p = ones(size(T, 2))
    q = M * p
    M .= Diagonal(1 ./ sqrt.(q)) * M
end

rescale!(T)

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

p = plot(sign ∘ μ, colormap=:redsblues);

savefig(p, "sv1.svg"); nothing # hide
```

![Second left singular measure](sv1.svg)

```julia
n_frames = 120
times = range(t₀, t₁, length=n_frames)

anim = @animate for t in times
    Φₜ(z) = Φ(z, t, τ, steps)

    F = BoxMap(:grid, Φₜ, domain, n_points=(6,6))
    F♯ = TransferOperator(F, S, S)
    rescale!(F♯)

    global maxiter, tol, v0
    U, σ, V = svds(F♯; maxiter=maxiter, tol=tol, v0=v0)

    μ = U[2]

    # do some rescaling to get a nice plot
    μ = ( x -> sign(x) * log(abs(x) + 1e-4) ) ∘ μ
    s = sign(μ[(10,40)])
    M = maximum(abs ∘ μ)
    μ = s/M * μ

    plot(μ, clims=(-1,1), colormap=:redsblues)
end;
gif(anim, "coherent.gif", fps=20)
```

![Coherent Sets](../assets/coherent.gif)


### References

[1] Froyland, G., Padberg-Gehle, K. (2014). Almost-Invariant and Finite-Time Coherent Sets: Directionality, Duration, and Diffusion. In: Bahsoun, W., Bose, C., Froyland, G. (eds) Ergodic Theory, Open Dynamics, and Coherent Structures. Springer Proceedings in Mathematics & Statistics, vol 70. Springer, New York, NY. https://doi.org/10.1007/978-1-4939-0419-8_9
