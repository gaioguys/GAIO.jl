# Cyclic Sets

### Mathematical Background

We can extend the idea of almost invariant sets to sets which are _cyclic_ in nature. We wish to find sets ``A_0, \ldots, A_{r-1}`` such that 
```math
A_{k \, \text{mod} \, r} \approx f^{-1} ( A_{k+1 \, \text{mod} \, r} ) , 
```
or in the context of the transfer operator, signed measures ``\mu_0, \ldots, \mu_{r-1}`` with 
```math
f_{\#}\,\mu_{k \, \text{mod} \, r} \approx \mu_{k+1 \, \text{mod} \, r} 
```
and supports on ``A_0, \ldots, A_{r-1}``, respectively. 

We can approximate a solution to this problem again as an eigenproblem, finding eigenmeasures ``\nu_0, \ldots, \nu_{r-1}`` corresponding to the ``r``-th roots of unity ``\omega_r^k = e^{2 \pi k / r},\ k = 0, \ldots, r-1``. We have a theorem from [complicated](@cite):

Suppose there exist sets ``A_0, \ldots, A_{r-1}\, \subset Q`` with ``A_{k \, \text{mod} \, r} \approx f^{-1} ( A_{k+1 \, \text{mod} \, r} )``. Then the rth power
```math
(f_{\#})^r = \underbrace{f_{\#} \circ \ldots \circ f_{\#}}_{r\ \text{times}}
```
has eigenvalue ``1`` with multiplicity at least ``r``. Further, there are ``r`` corresponding probability measures ``\mu_0, \ldots, \mu_{r-1}`` with supports on ``A_0, \ldots, A_{r-1}``, respectively. These ``\mu_k`` can be constructed from the eigenmeasures ``\nu_k`` of ``f_{\#}`` corresponding to the eigenvalues ``\omega_r^k`` as follows
```math
\mu_{\pi(l)} = \frac{1}{r} \sum_{k = 0}^{r-1} \omega_r^{k \cdot l} \nu_k
```
where ``\pi`` is some permutation of the indices ``\left\{ 0, \ldots, r \right\}``. 

### Example

Let us consider the map ``f : \mathbb{C} \to \mathbb{C}`` given by 
```math
f(z) = e^{-\frac{2 \pi i}{3}} \left( (|z|^2 + \alpha) z + \frac{1}{2} \bar{z}^2 \right) , 
```
with ``\alpha = -1.7``. To realize this in GAIO.jl, we will view ``\mathbb{C} \cong \mathbb{R}^2``. 

```@example 1
using GAIO

const α = -1.7
f(z) = exp(-2*pi*im/3) * ( (abs(z)^2 + α)*z + conj(z)^2 / 2 )
fr((x, y)) = reim( f(x + y*im) )

c, r = (0, 0), (1.5, 1.5)
Q = Box(c, r)
P = GridPartition(Q, (128,128))
F = BoxMap(fr, Q)

S = cover(P, (0,0))
W = unstable_set(F, S)
```

If we consider the chain recurrent set we notice that there seem to be some discrete "blobs". We may wonder how they interact. 

```@example 1
C = chain_recurrent_set(F, W, steps=2)
```

```@example 1
using Plots

p = plot(C)

savefig(p, "chain.svg"); nothing # hide
```

![Chain Recurrent Set](chain.svg)

```@example 1
T = TransferOperator(F, W, W)

# eigenvalues of Largest Magnitude (LM)
λ, ev = eigs(T; nev=32, which=:LM)

λ
```

```@example 1
p = scatter(λ)

savefig(p, "evs.svg"); nothing # hide
```

![Eigenvalues](evs.svg)

We see that the ``6``th roots of unity clearly seem to be part of the spectrum. We therefore can conclude that there is an approximate 6-cycle, and can extract the sets corresponding to the cycle. 

```@example 1
# by inspection we see that λ[1:6] are ω₆ᵏ, k = 0, ..., 5
ω = λ[1:6]
ν = ev[1:6]

# perform the sum described in the theorem
μ = [sum( 1/6 .* ω.^l .* ν ) for l in 0:5]

# grab the real components
# (they are all approximately real, but the data type is still ComplexF64)
μ = real .∘ μ

# threshhold to extract support of each μᵢ
# This depends on the result from ARPACK, 
# so it might be necessary to flip the `>`
τ = eps()
A = [
    BoxSet( P, Set(key for key in keys(μᵢ) if μᵢ[key] > τ) ) 
    for μᵢ in μ
]

# we won't rely on chance to get ARPACK right # hide
using Serialization # hide
A = deserialize("../assets/cyclic.ser") # hide
A
```

```@example 1
p = plot();
for (i, Aᵢ) in enumerate(A)
    global p;
    p = plot!(p, Aᵢ, color=i, fillalpha=0.6);
end

savefig(p, "supps.svg"); nothing # hide
```

![Cyclic Sets](supps.svg)

Note that we can also approximate the cyclic sets from the measures `μ` using sparse eigenbasis approximation (SEBA) as described in the corresponding section of the documentation. 
