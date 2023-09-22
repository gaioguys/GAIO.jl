Core.@doc raw"""
    seba(V::Vector{<:BoxFun}, Rinit=nothing; which=partition_disjoint, maxiter=5000) -> S, A

Construct a sparse eigenbasis approximation of `V`, as described in 
[1]. Returns an `Array` of `BoxFun`s corresponding to the eigenbasis, 
as well as a maximum-likelihood `BoxFun` that maps a box to the 
element of `S` which has the largest value over the support. 

The keyword `which` is used to set the threshholding heuristic, 
which is used to extract a partition of the supports from the 
sparse basis. Builtin options are
```julia
partition_unity, partition_disjoint, partition_likelihood
```
which are all exported functions. 
"""
function seba(V::AbstractArray{U}, Rinit=nothing; which=partition_disjoint) where {B,K,W,Q,D<:OrderedDict,U<:BoxFun{B,K,W,Q,D}}
    #supp = BoxSet(V[1])
    #all(x -> BoxSet(x) == supp, V) || throw(DomainError(V, "Supports of BoxFuns do not match."))
    supp = union((BoxSet(μ) for μ in V)...)

    V̄ = [μ[key] for key in supp.set, μ in V]
    S̄, R = seba(V̄, Rinit)
    S̄, Ā, τ = which(S̄)

    S = [BoxFun(supp, S̄[:, i]) for i in 1:size(S̄, 2)]
    A = BoxFun(supp, Ā)
    return S, A
end

function seba(V::AbstractArray{U}, Rinit=nothing; which=partition_unity) where {B,K,W,Q,D,U<:BoxFun{B,K,W,Q,D}}
    V = [
        BoxFun( V[i].partition, OrderedDict(V[i]) )
        for i in 1:length(V)
    ]   # convert to ordered collections to guarantee deterministic iteration order
    return seba(V, Rinit; which=which)
end

function partition_unity(S)
    S .= max.(S, 0)
    S_descend = sort(S, dims=2, rev=true)
    S_sum = cumsum(S_descend, dims=2)
    τᵖᵘ = maximum(S_descend[S_sum .> 1], init=zero(eltype(S)))
    S[S .≤ τᵖᵘ] .= 0
    A = argmax.(eachrow(S))
    return S, A, τᵖᵘ
end

function partition_disjoint(S)
    S .= max.(S, 0)
    S_descend = sort(S, dims=2, rev=true)
    τᵈᵖ = maximum(S_descend[:, 2], init=zero(eltype(S)))
    S[S .≤ τᵈᵖ] .= 0
    A = argmax.(eachrow(S))
    return S, A, τᵈᵖ
end

function partition_likelihood(S)
    A = argmax.(eachrow(S))
    M = S[:, A]
    A[M .≤ 0] .= 0
    S .= 0
    r = size(S, 2)
    for i in 1:r
        S[A .== i, i] .= M[A .== i]
    end
    return S, A, 0.
end

Core.@doc raw"""
    seba(V::Matrix{<:Real}, Rinit=nothing, maxiter=5000) -> S, R

Construct a sparse approximation of the basis `V`, as described in 
[1]. Returns matrices ``S``, ``R`` such that
```math
\frac{1}{2} \| V - SR \|_F^2 + \mu \| S \|_{1,1}
```
is minimized,
where ``\mu \in \mathbb{R}``, ``\| \cdot \|_F`` is the Frobenuius-norm, 
and ``\| \cdot \|_{1,1}`` is the element sum norm, and ``R`` 
is orthogonal. See [1] for further information on the argument 
`Rinit`, as well as a description of the algorithm. 

[1] Gary Froyland, Christopher P. Rock, and Konstantinos Sakellariou. 
Sparse eigenbasis approximation: multiple feature extraction 
across spatiotemporal scales with application to coherent set 
identification. Communications in Nonlinear Science and Numerical 
Simulation, 77:81-107, 2019. https://arxiv.org/abs/1812.02787
"""
function seba(V::AbstractArray{U}, Rinit=nothing, maxiter=5000) where {U}
    F = qr(V) # Enforce orthonormality
    V = Matrix(F.Q)
    p, r = size(V)
    μ = 0.99 / sqrt(p)

    S = zeros(size(V))
    # Perturb near-constant vectors
    for j = 1:r
        if maximum(V[:,j]) - minimum(V[:,j]) < 1e-14
            V[:,j] = V[:,j] .+ (rand(p, 1) .- 1 / 2) * 1e-12
        end
    end

    # Initialise rotation
    if Rinit ≡ nothing
        Rnew = Matrix(I, r, r)
    else
        # Ensure orthonormality of Rinit
        F = svd(Rinit)
        Rnew = F.U * F.Vt
    end

    R = zeros(r, r)
    iter = 0
    while norm(Rnew - R) > 1e-14 && iter < maxiter
        iter = iter + 1
        R = Rnew
        Z = V * R'
        # Threshold to solve sparse approximation problem
        for i = 1:r
            Si = sign.(Z[:,i]) .* max.(abs.(Z[:,i]) .- μ, zeros(p))
            S[:,i] = Si / norm(Si)
        end
        # Polar decomposition to solve Procrustes problem
        F = svd(S' * V, full=false)
        Rnew = F.U * F.Vt
    end

    # Choose correct parity of vectors and scale so largest value is 1
    for i = 1:r
        S[:,i] = S[:,i] * sign(sum(S[:,i]))
        S[:,i] = S[:,i] / maximum(S[:,i])
    end

    # Sort so that most reliable vectors appear first
    ind = sortperm(vec(minimum(S, dims=1)), rev=true)
    S = S[:, ind]

    return S, R
end
