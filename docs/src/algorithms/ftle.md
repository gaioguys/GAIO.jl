# Finite Time Lyapunov Exponents

### Mathematical Background
We change focus now to a continuous dynamical system, e.g. an ODE ``\dot{u} = g(t, u)`` with solution ``\Phi^{t,t_0} (x)``. Since ``\Phi^{t,t_0} (x)`` is continuously dependent on the initial condition ``x``, there exists an ``\tilde{x}`` near ``x`` with ``sup_{t \in [t_0 , t_0 + T]} \| \Phi^{t,t_0} (\tilde{x}) - \Phi^{t,t_0} (x) \| < \epsilon`` for any fixed ``\epsilon > 0`` and ``T`` small enough. We wish to characterize this expansion term. We write ``\Phi = \Phi^{t_0+T,t_0}`` and ``y = x + \delta x_0`` where ``\delta x_0 \in \mathbb{R}^d`` is infinitesimal. Then if ``g`` is ``\mathcal{C}^1`` w.r.t. ``x``,
```math
\delta x (t_0 + T) := \Phi (y) - \Phi (x)
= D_x \Phi (x) \cdot \delta x_0 + \mathcal{O}(\| \delta x_0 \|^2)
```
Hence we can write 
```math
\| \delta x (t_0 + T) \|_2 = \| D_x \Phi (x) \cdot \delta x_0 \|_2 \leq \| D_x \Phi (x) \|_2 \cdot \| \delta x_0 \|_2
```
or equivalently
```math
\frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \leq \| D_x \Phi (x) \|_2
```
where equality holds if ``\delta x_0`` is the eigenvector corresponding to the largest eigenvalue of 
```math
\Delta = \left( D_x \Phi (x) \right)^T \left( D_x \Phi (x) \right) . 
```
Hence if we define 
```math
\sigma (x) = \sigma^{t_0 + T, t_0} (x) := \frac{1}{T} \ln \left( \sqrt{\lambda_{\text{max}}} (\Delta) \right) = \frac{1}{T} \ln \left( \sup_{\delta x_0} \frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \right)
```
then 
```math
\| \delta x (t_0 + T) \|_2 \leq e^{T \cdot \sigma (x)} \cdot \| \delta x_0 \|_2 . 
```
From this we see why ``\sigma (x)`` is called the _maximal finite-time lyapunov exponent (FTLE)_. 

The definition of ``\sigma (x)`` leads to a natural _ansatz_ for approximating the FTLE: compute ``\frac{1}{T} \ln \left( \sup_{\delta x_0} \frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \right)`` for each of a set of test points ``\| \delta x_0 \|`` of fixed order ``\epsilon > 0`` and set ``\sigma (x)`` to be the maximum over this set of test points. 

An extension of this technique can be made for _ergodic_ systems, as shown in [beyn](@cite): 

when calculating the maximal Lyapunov exponent for a discrete dynamical system ``x_{n+1} = f(x_k)`` defined as 
```math
\lambda (x, v) = \lim_{n \to \infty} \frac{1}{n} \log \| Df^n (x) \cdot v \|
```
a known technique is to use a QR iteration. Let ``A = Q(A) R(A)`` be the unique QR-decomposition of a nonsingular matrix ``A`` into an orthogonal matrix ``Q(A)`` and an upper-triangular matrix ``R(A)``. Then from ``\| Av \| = \| R(A) v \|`` we have 
```math
\lim_{n \to \infty} \frac{1}{n} \log \| Df^n(x) v \| = \lim_{n \to \infty} \frac{1}{n} \log \| R(Df^n(x))v \|. 
```
Further, the Lyapunov exponents of the system ``\lambda_1, \ldots, \lambda_d`` (which are costant over the phase space for an ergodic system) can be found via
```math
\lambda_j = \lim_{n \to \infty} \frac{1}{n} \log R_{jj}(Df^n(x))
```
the ``j``-th diagonal element of ``R``, for ``j = 1, \ldots, d``. 

Using an extension of the Birkhoff ergodic theorem it can be proven that this method is equivalent to computing
```math
\lambda_j = \lim_{n \to \infty} \int \log R_{jj}(Df^n(x)) \, d\mu
```
where ``\mu`` is a measure which is ergodic and invariant under ``f``. 

```@docs; canonical=false
finite_time_lyapunov_exponents
```

### Example

We will continue using the periodically driven double-gyre introduced in the section on [Almost Invariant (metastable) Sets](@ref). See that code block for the definition of the map. 

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
P = GridPartition(domain, (256, 128))
S = cover(P, :)

F = BoxMap(:grid, Φₜ₀ᵗ¹, domain, n_points=(6,6))

γ = finite_time_lyapunov_exponents(F, S; T=Tspan)
```

```@example 1
using Plots

p = plot(γ, clims=(0,2), colormap=:jet);

savefig("ftle1.svg"); nothing # hide
```

![FTLE field at time 0](ftle1.svg)

Since this map is time-dependent, the FTLE field will change over time as well. 

```julia
n_frames = 120
times = range(t₀, t₁, length=n_frames)

anim = @animate for t in times
    Φₜ(z) = Φ(z, t, τ, steps)

    F = BoxMap(:grid, Φₜ, domain, n_points=(6,6))
    γ = finite_time_lyapunov_exponents(F, S; T=Tspan)

    plot(γ, clims=(0,2), colormap=:jet)
end;
gif(anim, "ftle1.gif", fps=20)
```

![FTLE field](../assets/ftle1.gif)

### Example 2: An Ergodic System

```julia
using GAIO
using Plots
using StaticArrays

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = SA{Float64}[ 1 - a*x^2 + y, b*x ]
Df((x,y)) = SA{Float64}[-2*a*x    1.;
                         b        0.]

center, radius = (0, 0), (3, 3)
P = GridPartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 20)

T = TransferOperator(F, A, A)
(λ, ev) = eigs(T)
μ = real ∘ ev[1]

σ16 = finite_time_lyapunov_exponents(f, Df, μ, n=16)
σ8  = finite_time_lyapunov_exponents(f, Df, μ, n=8)

σ = 2*σ16 - σ8
```
