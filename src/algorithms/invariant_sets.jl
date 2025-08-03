Core.@doc raw"""
    preimage(F::BoxMap, B::BoxSet, Q::BoxSet) -> BoxSet

Compute the (restricted to `Q`) preimage of `B` under `F`, i.e.
```math
F^{-1} (B) \cap Q . 
```
Note that the larger ``Q`` is, the more calculation time required. 
"""
function preimage(F::BoxMap, B::BoxSet, Q::BoxSet)
    T = TransferOperator(F, Q, B)
    μ = BoxMeasure(B)
    return BoxSet(T'μ)
end

#=
function preimage(F::BoxMap, B::BoxSet, Q::BoxSet)
    T = TransferOperator(F, Q, B)
    if B == Q
        C⁻ = vec( sum(T.mat, dims=1) .> 0 ) # C⁻ = B ∩ F⁻¹(B)
        return BoxSet(T.domain, C⁻)     # same result but faster
    else
        μ = BoxMeasure(B)
        return BoxSet(T'μ)
    end
end
=#

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
μ = BoxMeasure(B)

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
    iterate_until_equal(f, start; max_iterations = Inf)

Iterate a function `f` starting at `start` until 
a fixed point is reached or `max_iterations` 
have been performed. 
One iteration is always guaranteed! 
"""
function iterate_until_equal(f, start; max_iterations=Inf)
    state, state_old = f(start), start
    n_iterations = 1

    while state ≠ state_old  &&  n_iterations < max_iterations
        state_old = state
        state = f(state)
        n_iterations += 1
    end

    @debug "number of iterations" n_iterations max_iterations
    return state
end

"""
    ω(F::BoxMap, B::BoxSet; subdivision=true, steps=subdivision ? 12 : 64) -> BoxSet

Compute the ω-limit set of `B` under `F`. 
"""
function ω(F, B::BoxSet; 
            subdivision=true, steps=subdivision ? 12 : 64)
    iter = (S -> S ∩ F(S)) ∘ (subdivision ? subdivide : identity)
    return iterate_until_equal(iter, B; max_iterations=steps)
end

"""
    ω(F::BoxMap, B::BoxSet; subdivision=true, steps=subdivision ? 12 : 64) -> BoxSet

Compute the α-limit set of `B` under `F`. 
"""
function α(F, B::BoxSet; 
            subdivision=true, steps=subdivision ? 12 : 64)
    F⁻¹(S) = preimage(F, S, S)  # computes  F⁻¹(S) ∩ S  which is all we need
    return ω(F⁻¹, B; subdivision=subdivision, steps=steps)
end

const maximal_backward_invariant_set = ω
const relative_attractor = ω
const maximal_forward_invariant_set = α

"""
    maximal_invariant_set(F::BoxMap, B::BoxSet; subdivision=true, steps=subdivision ? 12 : 64) -> BoxSet

Compute the maximal invariant set of `F` 
within the set `B`. 
"""
function maximal_invariant_set(F, B::BoxSet; 
                                subdivision=true, steps=subdivision ? 12 : 64)
    G(S) = F(S) ∩ preimage(F, S, S)  # computes  F⁻¹(S) ∩ S ∩ F(S)
    return ω(G, B; subdivision=subdivision, steps=steps)
end

"""
    recurrent_set(F::BoxMap, B::BoxSet; subdivision=true, steps=subdivision ? 12 : 64) -> BoxSet

Compute the (chain) recurrent set within the box set `B`. 
"""
function recurrent_set(F::BoxMap, B::BoxSet; 
                        subdivision=true, steps=subdivision ? 12 : 64)
    iter = (S -> morse_sets(F, S)) ∘ (subdivision ? subdivide : identity)
    return iterate_until_equal(iter, B; max_iterations=steps)
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


#=
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


# More efficient (but ugly) versions of invariant set algorithms

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
