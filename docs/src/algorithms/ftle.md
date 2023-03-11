# Finite Time Lyapunov Exponents

### Mathematical Background
We change focus now to a continuous dynamical system, e.g. an ODE ``\dot{u} = g(t, u)`` with solution ``\Phi^{t,t_0} (x)``. Since ``\Phi^{t,t_0} (x)`` is continuously dependent on the initial condition ``x``, there exists an ``\tilde{x}`` near ``x`` with ``sup_{t \in [t_0 , t_0 + T]} \| \Phi^{t,t_0} (\tilde{x}) - \Phi^{t,t_0} (x) \| < \epsilon`` for any fixed ``\epsilon > 0`` and ``T`` small enough. We wish to characterize this expansion term. We write ``y = x + \delta x_0`` where ``\delta x_0 \in \mathbb{R}^d`` is infinitesimal. Then if ``g`` is ``\mathcal{C}^1`` w.r.t. ``x``,
```math
\delta x (t_0 + T) := \Phi^{t_0 + T, t_0} (y) - \Phi^{t_0 + T, t_0} (x)
= D_x \Phi^{t_0 + T, t_0} (x) \cdot \delta x_0 + \mathcal{O}(\| \delta x_0 \|^2)
```
Hence we can write 
```math
\| \delta x (t_0 + T) \|_2 = \| D_x \Phi^{t_0 + T, t_0} (x) \cdot \delta x_0 \|_2 \leq \| D_x \Phi^{t_0 + T, t_0} (x) \|_2 \cdot \| \delta x_0 \|_2
```
or equivalently
```math
\frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \leq \| D_x \Phi^{t_0 + T, t_0} (x) \|_2
```
where equality holds if ``\delta x_0`` is the eigenvector corresponding to the largest eigenvalue of 
```math
\Delta = \left( D_x \Phi^{t_0 + T, t_0} (x) \right)^T \left( D_x \Phi^{t_0 + T, t_0} (x) \right) . 
```
Hence if we define 
```math
\sigma^{t_0 + T, t_0} (x) = \frac{1}{T} \ln \left( \sqrt{\lambda_{\text{max}}} (\Delta) \right) = \frac{1}{T} \ln \left( \sup_{\delta x_0} \frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \right)
```
then 
```math
\| \delta x (t_0 + T) \|_2 \leq e^{T \cdot \sigma^{t_0 + T, t_0} (x)} \cdot \| \delta x_0 \|_2 . 
```
From this we see why ``\sigma^{t_0 + T, t_0} (x)`` is called the _maximal finite-time lyapunov exponent (FTLE)_. 

The definition of ``\sigma^{t_0 + T, t_0} (x)`` leads to a natural _ansatz_ for approximating the FTLE: compute ``\frac{1}{T} \ln \left( \sup_{\delta x_0} \frac{ \| \delta x (t_0 + T) \|_2 }{ \| \delta x_0 \|_2 } \right)`` for each of a set of test points ``\| \delta x_0 \|`` of fixed order ``\epsilon > 0`` and set ``\sigma^{t_0 + T, t_0} (x)`` to be the maximum over this set of test points. 

An extension of this technique can be made for _ergodic_ systems, as shown in [1]: 

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

```@docs
finite_time_lyapunov_exponents
```

### Example

```julia
using StaticArrays

# an ergodic transformation
f((x, y)) = SA[x + y, x + 2y] .% 1
Df((x, y)) = SA[1.   1.;
                1.   2.]

Q = Box((0.5, 0.5), (0.5, 0.5))
P = BoxPartition(Q, (1000,1000))
S = cover(P, :)

F = BoxMap(f, Q)
T = TransferOperator(F, S, S)
λ, ev = eigs(T)   # The Lebesque measure - i.e. the constant-weight measure - is invariant
μ = abs ∘ ev[1]

σ8 = finite_time_lyapunov_exponents(f, Df, μ; n = 8)
σ16 = finite_time_lyapunov_exponents(f, Df, μ; n = 16)

σ = 2*σ16 - σ8
```

### References

[1] Beyn, WJ., Lust, A. A hybrid method for computing Lyapunov exponents. _Numer. Math._ 113, 357–375 (2009). https://doi.org/10.1007/s00211-009-0236-4
