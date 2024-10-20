# Sparse Eigenbasis Approximation (SEBA)

### Mathematical Background

A common use pattern in GAIO.jl is to construct the transfer operator (or related operators), and then investigate the eigenfunctions for some specific structure. This may be e.g. partitioning the state space based on the result of the second leading eigenvector to find almost invariant sets. A common approach is to use ``k``-means, though this may not be entirely sufficient. The method of Froyland et. al. [seba](@cite) attempts to find a sparse basis that approximately spans the one produced by a set of eigenvalues. 

More specifically: denote some eigenvalues ``\lambda_1 \geq \ldots \geq \lambda_r`` and corresponding eigenvectors ``v_1, \ldots, v_r \in \mathbb{R}^d`` (typically ``r \ll p``) of a data matrix. We write ``V = [v_1 \vert \ldots \vert v_r]``. In the context of GAIO.jl this matrix may be the discretized transfer operator. The eigenvectors span a basis ``\mathcal{V} \subset \mathbb{R}^p``. We wish to transform this basis into a basis of sparse vectors ``s_1, \ldots, s_r \in \mathbb{R}^p`` that span a subspace ``\mathcal{S} \approx \mathcal{V}``. Mathematically, this can be formulated as solving the optimization problem
```math
\underset{S, R}{\mathrm{arg\,min}}\ \frac{1}{2} \| V - S R \|_F^2 + \mu \| S \|_{1,1}
```
where ``S \in \mathbb{R}^{n \times p}`` has ``\ell_2`` norm in each column, ``R \in \mathbb{R}^{r \times r}`` is orthogonal, and ``\mu > 0`` is a penalty term. ``\| \cdot \|_F`` denotes the Frobenius norm and ``\| \cdot \|_{1,1}`` the element sum norm. 

Solving this problem for ``S = [s_1 \vert \ldots \vert s_r]`` is done by the SEBA algorithm [1], (which is based on sparse principal component analysis by rotation and truncation - _SPCArt_ [hu2014sparse](@cite)). At this point, most of the work is finished. Indeed, one may be satisfied with the sparse basis alone. However, recall that the goal is to _partition_ the state space into sets based on the eigenvalues. Hence the final step is to threshhold the sparse vectors to fix which indices are in or out of a feature, that is, find an appropriate ``\tau`` and set `S[S .≤ τ] .= 0` such that the least infomation is lost. For this, three heuristics are offered by GAIO.jl:
* Maximum likelihood partition (without threshholding): For each feature (each row) ``i``, set ``S_{ij} = 0`` for all ``j`` except ``j_0 = \underset{j}{\mathrm{arg\,min}}\ S_{ij}``. 
* Hard partition: For each feature (each row) ``i``, write the values ``s_{i1}, \ldots, s_{ir}`` of ``S_{i\cdot}`` in decreasing order. Choose the threshhold ``\tau^{dp} = \underset{1 \leq i \leq p}{\mathrm{max}}\ s_{i2}``, i.e. the maximum over the _second largest_ element of each row. Set `S[S .≤ τ] .= 0`. 
* Partition of unity: For each feature (each row) ``i``, write the values ``s_{i1}, \ldots, s_{ir}`` of ``S_{i\cdot}`` in decreasing order. Choose the threshhold ``\tau^{pu} = \underset{1 \leq i \leq p,\  1 \leq j \leq r}{\mathrm{max}} \left\{ s_{ij} \vert \sum_{k=1}^j s_{ik} > 1 \right\}``, i.e. the minimum threshhold such that all rows sum to less than ``1``. Set `S[S .≤ τ] .= 0`. 
Note that the final heuristic does not return a strict partition of the features, but rather a partition of unity. 

By default when calling GAIO.jl's `seba`, hard partitioning is performed. 

```@docs; canonical=false
seba
```

### Example

We will continue using the periodically driven double-gyre introduced in the section on [Almost Invariant (metastable) Sets](@ref). See that code block for the definition of the map. 

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

F = BoxMap(:montecarlo, Φₜ₀ᵗ¹, domain, n_points=32)
F = BoxMap(:pointdiscretized, Φₜ₀ᵗ¹, domain, test_points) # hide

T = TransferOperator(F, S, S)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(T, 2))
λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

μ = abs ∘ ev[2]
```

```@example 1
using Plots

p = plot(μ, colormap=:jet);

savefig("second_eigvec.svg"); nothing # hide
```

![Second egienvector of the transfer operator](second_eigvec.svg)

We notice there are two "blobs" defining the second eigenmeasure. These correspond to the almost invariant sets; there are two "vortices" where mass flows in a circular pattern and doesn't mix with the rest of the domain. We wish to isolate these blobs using `seba`

```@example 1
# seba expects real numbers, ev is complex, so we grab the real components. 
# We also potentially have to scale by -1, this depends on what Arpack 
# returns so always try both
re_ev = real .∘ (-1 .* ev)

ev_seba, feature_vec = seba(re_ev, which=partition_unity)
μ1, μ2 = ev_seba[1], ev_seba[2]

S1 = BoxSet(P, Set(key for key in keys(μ1) if μ1[key] > 0.01))
S2 = BoxSet(P, Set(key for key in keys(μ2) if μ2[key] > 0.01))

setdiff!(S1, S2) # hide
setdiff!(S2, S1) # hide

p = plot(S1, xlims=(0,2), ylims=(0,1), color=:red);
p = plot!(p, S2, color=:blue);

savefig("seba.svg"); nothing # hide
```

![Almost invriant sets isolated by SEBA](seba.svg)
