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
    C⁻ = vec( sum(P.mat, dims=1) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
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
    C⁺ = vec( sum(P.mat, dims=2) .> 0 ) # C⁺ = B ∩ F(B)
    C⁻ = vec( sum(P.mat, dims=1) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
    C  = C⁺ .& C⁻   # C  =  C⁺ ∩ C⁻  =  F(B) ∩ B ∩ F⁻¹(B)
    return BoxSet(P.domain, C)
end

"""
    relative_attractor(F::BoxMap, B::BoxSet; steps=12) -> BoxSet
    maximal_backward_invariant_set(F::BoxMap, B::BoxSet; steps=12) -> BoxSet

Compute the attractor relative to `B`. `B` should be 
a (coarse) covering of the relative attractor, e.g. 
`B = cover(P, :)` for a partition `P`.
"""
function maximal_backward_invariant_set end

const relative_attractor = maximal_backward_invariant_set

"""
    maximal_forward_invariant_set(F::BoxMap, B::BoxSet; steps=12)

Compute the maximal forward invariant set contained in `B`. 
`B` should be a (coarse) covering of a forward invariant set, 
e.g. `B = cover(P, :)` for a partition `P`.
"""
function maximal_forward_invariant_set end

restricted_image(F, B) = F(B) ∩ B

"""
    maximal_invariant_set(F::BoxMap, B::BoxSet; steps=12)

Compute the maximal invariant set contained in `B`. 
`B` should be a (coarse) covering of an invariant set, 
e.g. `B = cover(P, :)` for a partition `P`.
"""
function maximal_invariant_set end


for (algorithm, func) in [
        :maximal_forward_invariant_set  => preimage,
        :maximal_backward_invariant_set => restricted_image,
        :maximal_invariant_set          => symmetric_image
    ]

    @eval function $(algorithm)(
            F::BoxMap, B₀::BoxSet{Box{N,T}}, subdivision::Val{true}; steps=12
        ) where {N,T}

        H(B) = $(func)(F, B)
        B = copy(B₀)
        for k in 1:steps
            B = subdivide(B, (k % N) + 1)
            B = H(B)
        end
        return B
    end

    @eval function $(algorithm)(
            F::BoxMap, B₀::BoxSet{Box{N,T}}, subdivision::Val{false}; steps=12, 
        ) where {N,T}

        H(B) = $(func)(F, B)
        B = copy(B₀)
        for k in 1:steps
            C = H(B)
            B == C && break
            B = C
        end
        return B
    end

    @eval function $(algorithm)(
            F::BoxMap, B₀::BoxSet{Box{N,T}};
            steps=12, subdivision::Bool=true
        ) where {N,T}

        $(algorithm)(F, B₀, Val(subdivision); steps=steps)
    end

end

#=
"""
helper function to compute intersections
"""
⊓(a, b) = a != 0 && b != 0

for (algorithm, func) in [
        maximal_forward_invariant_set  => identity,
        maximal_backward_invariant_set => transpose
    ]

    @eval function $(algorithm)(
            F♯::TransferOperator, v⁺::AbstractVector, v⁻::AbstractVector; 
            steps=12, subdivision::Val{false}
        )

        fill!(v⁺, 1); fill(v⁻, 0)
        M = $func(F♯.mat)
        for _ in 1:steps
            v⁺ == v⁻ && break
            v⁻ .= v⁺
            v⁺  = mul!(v⁺, M, v⁻)
            v⁺ .= v⁺ .⊓ v⁻
        end
        return BoxSet(F♯.domain, v⁺ .!= 0)
    end

    @eval function $(algorithm)(
            F::BoxMap, B::BoxSet{Box{N,T}}, v⁺::AbstractVector, v⁻::AbstractVector; 
            steps=12, subdivision::Val{false}
        ) where {N,T}

        F♯ = TransferOperator(F, B, B)
        resize!(v⁺, length(B)); resize!(v⁻, length(B))
        return $(algorithm)(F♯, v⁺, v⁻; steps=steps, subdivision=subdivision)
    end

    @eval function $(algorithm)(
            F::BoxMap, B::BoxSet{Box{N,T}}; 
            steps=12, subdivision::Val{false}
        ) where {N,T}
        
        return $(algorithm)(F, B, Float64[], Float64[]; steps=steps, subdivision=subdivision)
    end

end

function maximal_invariant_set(
        F♯::TransferOperator, 
        v⁺::AbstractVector, v⁰::AbstractVector, v⁻::AbstractVector;
        steps=12, subdivision::Val{false}
    )

    fill!(v⁻, 1); fill!(v⁰, 0); fill!(v⁺, 0)
    M = F♯.mat
    for _ in 1:steps
        v⁻ == v⁺ && break
        v⁺ .= v⁻
        v⁻ .= mul!(v⁻, M, v⁺)
        v⁰ .= mul!(v⁰, M, v⁺)
        v⁻ .= v⁺ .⊓ v⁻
        v⁻ .= v⁰ .⊓ v⁻
    end
    return BoxSet(F♯.domain, v⁻ .!= 0)
end

function maximal_invariant_set(
        F::BoxMap, B::BoxSet{Box{N,T}}, 
        v⁺::AbstractVector, v⁰::AbstractVector, v⁻::AbstractVector;
        steps=12, subdivision::Val{false}
    ) where {N,T}

    F♯ = TransferOperator(F, B, B)
    resize!(v⁺, length(B)); resize!(v⁰, length(B)); resize!(v⁻, length(B))
    maximal_invariant_set(F♯, v⁺, v⁰, v⁻; steps=steps, subdivision=subdivision)
end

function maximal_invariant_set(
        F::BoxMap, B::BoxSet{Box{N,T}}; 
        steps=12, subdivision::Val{false}
    ) where {N,T}

    maximal_invariant_set(F, B, Float64[], Float64[], Float64[], steps=steps, subdivision=subdivision)
end
=#
