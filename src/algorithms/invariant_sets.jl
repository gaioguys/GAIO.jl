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
        B   = subdivide(B, (k % N) + 1)
        F♯  = TransferOperator(F, B, B)
        G   = MatrixNetwork(F♯)
        SCC = scomponents(G)
        B   = BoxSet(morse_tiles(F♯, SCC))
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
    P  = TransferOperator(F, B, B)
    C⁻ = vec( sum(P.mat, dims=2) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
    return BoxSet(P.domain, C⁻)
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
    P  = TransferOperator(F, B, B)
    C⁺ = vec( sum(P.mat, dims=1) .> 0 ) # C⁺ = B ∩ F(B)
    C⁻ = vec( sum(P.mat, dims=2) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
    C  = C⁺ .& C⁻   # C  =  C⁺ ∩ C⁻  =  F(B) ∩ B ∩ F⁻¹(B)
    return BoxSet(P.domain, C)
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
