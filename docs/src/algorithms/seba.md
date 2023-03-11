# Sparse Eigenbasis Approximation (SEBA)

### Mathematical Background

A common use pattern in GAIO.jl is to construct the transfer operator (or related operators), and then investigate the eigenfunctions for some specific structure. This may be e.g. partitioning the state space based on the result of the second leading eigenvector to find almost invariant sets. A common approach is to use ``k``-means, though this may not be entirely sufficient. The method of Froyland et. al. [1] attempts to find a sparse basis that approximately spans the one produced by a set of eigenvalues. 

More specifically: denote some eigenvalues ``\lambda_1 \geq \ldots \geq \lambda_r`` and corresponding eigenvectors ``v_1, \ldots, v_r \in \mathbb{R}^d`` (typically ``r \ll p``) of a data matrix. We write ``V = [v_1 \vert \ldots \vert v_r]``. In the context of GAIO.jl this matrix may be the discretized transfer operator. The eigenvectors span a basis ``\mathcal{V} \subset \mathbb{R}^p``. We wish to transform this basis into a basis of sparse vectors ``s_1, \ldots, s_r \in \mathbb{R}^p`` that span a subspace ``\mathcal{S} \approx \mathcal{V}``. Mathematically, this can be formulated as solving the optimization problem
```math
\underset{S, R}{\mathrm{arg\,min}}\ \frac{1}{2} \| V - S R \|_F^2 + \mu \| S \|_{1,1}
```
where ``S \in \mathbb{R}^{n \times p}`` has ``\ell_2`` norm in each column, ``R \in \mathbb{R}^{r \times r}`` is orthogonal, and ``\mu > 0`` is a penalty term. ``\| \cdot \|_F`` denotes the Frobenius norm and ``\| \cdot \|_{1,1}`` the element sum norm. 

Solving this problem for ``S = [s_1 \vert \ldots \vert s_r]`` is done by the SEBA algorithm [1], (which is based on sparse principal component analysis by rotation and truncation - _SPCArt_ [2]). At this point, most of the work is finished. Indeed, one may be satisfied with the sparse basis alone. However, recall that the goal is to _partition_ the state space into sets based on the eigenvalues. Hence the final step is to threshhold the sparse vectors to fix which indices are in or out of a feature, that is, find an appropriate ``\tau`` and set `S[S .≤ τ] .= 0` such that the least infomation is lost. For this, three heuristics are offered by GAIO.jl:
* Maximum likelihood partition (without threshholding): For each feature (each row) ``i``, set ``S_{ij} = 0`` for all ``j`` except ``j_0 = \underset{j}{\mathrm{arg\,min}}\ S_{ij}``. 
* Hard partition: For each feature (each row) ``i``, write the values ``s_{i1}, \ldots, s_{ir}`` of ``S_{i\cdot}`` in decreasing order. Choose the threshhold ``\tau^{dp} = \underset{1 \leq i \leq p}{\mathrm{max}}\ s_{i2}``, i.e. the maximum over the _second largest_ element of each row. Set `S[S .≤ τ] .= 0`. 
* Partition of unity: For each feature (each row) ``i``, write the values ``s_{i1}, \ldots, s_{ir}`` of ``S_{i\cdot}`` in decreasing order. Choose the threshhold ``\tau^{pu} = \underset{1 \leq i \leq p,\  1 \leq j \leq r}{\mathrm{max}} \left\{ s_{ij} \vert \sum_{k=1}^j s_{ik} > 1 \right\}``, i.e. the minimum threshhold such that all rows sum to less than ``1``. Set `S[S .≤ τ] .= 0`. 
Note that the final heuristic does not return a strict partition of the features, but rather a partition of unity. 

By default when calling GAIO.jl's `seba`, hard partitioning is performed. 

```@docs
seba
```

### Example

```julia
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

t₀, τ, t₁ = 0, 0.1, 2
Φₜ₀ᵗ¹(z) = Φ(z, t₀, τ, t₁)

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxPartition(domain, (256, 128))
S = cover(P, :)
F = BoxMap(:montecarlo, Φₜ₀ᵗ¹, domain)
T = TransferOperator(F, S, S)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(T, 2))
λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

ev_seba = seba(ev, which=partition_unity)

using Plots
p1 = plot(ev_seba[1])
p2 = plot(ev_seba[2])
plot(p1, p2, layout=2, size=(1200,600))
```

### References

[1] Gary Froyland, Christopher P. Rock, and Konstantinos Sakellariou. Sparse eigenbasis approximation: multiple feature extraction across spatiotemporal scales with application to coherent set identification. Communications in Nonlinear Science and Numerical Simulation, 77:81-107, 2019. https://arxiv.org/abs/1812.02787

[2] Z. Hu, G. Pan, Y. Wang, and Z. Wu. Sparse principal component analysis via rotation and truncation. IEEE Transactions on Neural Networks and Learning Systems, 27(4):875–890, 2016.
