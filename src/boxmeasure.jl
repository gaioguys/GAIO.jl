"""
    BoxMeasure(partition, vals)

Discretization of a measure over the domain `partition.domain`,
as a piecewise constant function over the boxes of `partition`. 
    
Implemented as a sparse vector over the indices of `partition`. 

Constructors:
* BoxMeasure with constant weight 0 of Type `T` (default Float64) 
supported over a `BoxSet` `B`:
```julia
μ = BoxMeasure(B, T)
```
* BoxMeasure with specified weights per key
```julia
P = B.partition
weights = Dict( key => 1 for key in keys(B) )
BoxMeasure(P, weights)
```
* BoxMeasure with vector of weights supportted over a `BoxSet` `B`: 
```julia
weights = rand(length(B))
μ = BoxMeasure(B, weights)
```
(Note that since `Boxset`s do not have a deterministic iteration 
order by default, this may have unintented results. This 
constructor should therefore only be used with 
`BoxSet{<:Any, <:Any, <:OrderedSet}` types)

Fields:
* `partition`: An `AbstractBoxPartition` whose indices are used 
for `vals`
* `vals`: A dictionary whose keys are the box indices from 
`partition`, and whose values represent the values of the function. 

Methods implemented:

    length, sum, iterate, values, isapprox, ∘, LinearAlgebra.norm, LinearAlgebra.normalize!

!!! note "Norms of BoxMeasures"
    The p-norm of a BoxMeasure is the L^p function norm 
    of its density (w.r.t. the Lebesgue measure). 


"""
struct BoxMeasure{B,K,V,P<:AbstractBoxPartition{B},D<:AbstractDict{K,V}} <: AbstractVector{V}
    partition::P
    vals::D
end

BoxMeasure(boxset::BoxSet, vals, dicttype=OrderedDict) = BoxMeasure(boxset.partition, dicttype(zip(boxset.set, vals)))
BoxMeasure(boxset::BoxSet, T::Type, dicttype=OrderedDict) = BoxMeasure(boxset.partition, dicttype(key=>zero(T) for key in boxset.set))
BoxMeasure(boxset::BoxSet{B}) where {N,T,B<:Box{N,T}} = BoxMeasure(boxset, T)
BoxMeasure(boxmeas::BoxMeasure, vals, dicttype=OrderedDict)= BoxMeasure(boxmeas.partition, dicttype(zip( keys(boxmeas), vals )))

BoxSet(boxmeas::BoxMeasure; settype=OrderedSet) = BoxSet(boxmeas.partition, settype(keys(boxmeas)))

Base.Dict(boxmeas::BoxMeasure) = Dict( zip( keys(boxmeas), values(boxmeas) ) )
OrderedCollections.OrderedDict(boxmeas::BoxMeasure) = OrderedDict( zip( keys(boxmeas), values(boxmeas) ) )

box_pairs(fun::BoxMeasure) = (key_to_box(fun.partition, key) => weight for (key,weight) in fun.vals)

Core.@doc raw"""
    sum(f, μ::BoxMeasure)
    sum(f, μ::BoxMeasure, B::BoxSet)
    μ(B) = sum(x->1, μ, B)

Integrate a function `f` with respect to the measure `μ`, that is,
if `boxmeas` is the discretization of a measure ``\mu`` over the domain 
``Q``, then approximate the value of 
```math
\int_Q f \, d\mu .
```
If a BoxSet `B` is passed as the third argument, then the 
integration is restricted to the boxes in `B`
```math
\int_{Q \cap \bigcup_{b \in B} b} f \, d\mu .
```
The notation `μ(B)` is offered to compute 
``\mu (\bigcup_{b \in B} b)``. 
"""
function Base.sum(f, boxmeas::BoxMeasure{B,K,V,P,D}; init...) where {B,K,V,P,D}
    sum(box_pairs(boxmeas); init...) do pair
        box, val = pair
        f(box.center) * val
    end
end

function Base.sum(f, boxmeas::BoxMeasure{B,K,V,P,D}, boxset::Union{Box,BoxSet}; init...) where {B,K,V,P,D}
    support = cover(boxmeas.partition, boxset)
    boxmeas_new = BoxMeasure(
        boxmeas.partition, 
        D((key=>val) for (key,val) in boxmeas.vals if key in support.set)
    )
    sum(f, boxmeas_new; init...)
end

(boxmeas::BoxMeasure)(boxset::Union{Box,BoxSet}) = sum(_->1, boxmeas, boxset)

function Base.show(io::IO, g::BoxMeasure)
    print(io, "BoxMeasure in $(g.partition) with $(length(g.vals)) stored weights")
end

Base.length(fun::BoxMeasure) = length(fun.vals)
Base.size(fun::BoxMeasure) = (length(fun),)
Base.keytype(::BoxMeasure{B,K,V}) where {B,K,V} = K
Base.eltype(::BoxMeasure{B,K,V}) where {B,K,V} = V
Base.keys(fun::BoxMeasure) = keys(fun.vals)
Base.values(fun::BoxMeasure) = values(fun.vals)
Base.pairs(fun::BoxMeasure) = pairs(fun.vals)
Base.show(io::IO, ::MIME"text/plain", fun::BoxMeasure) = show(io, fun)
Base.maximum(fun::BoxMeasure) = maximum(values(fun))
Base.minimum(fun::BoxMeasure) = minimum(values(fun))

function Base.iterate(boxmeas::BoxMeasure, i...)
    itr = iterate(boxmeas.vals, i...)
    isnothing(itr) && return itr
    ((key, val), j) = itr
    box = key_to_box(boxmeas.partition, key)
    return ((box => val), j)
end

function LinearAlgebra.norm(boxmeas::BoxMeasure, p::Real=2) 
    norm((val / volume(box)^(1/p) for (box,val) in boxmeas), p)
end

function LinearAlgebra.normalize!(boxmeas::BoxMeasure)
    λ = inv(norm(boxmeas))
    map!(x -> λ*x, values(boxmeas.vals))
    return boxmeas
end

Base.getindex(boxmeas::BoxMeasure{B,K,V}, key::Vararg{<:Integer,N}) where {N,B<:Box{N},K,V} = get(boxmeas.vals, key, zero(V))
Base.getindex(boxmeas::BoxMeasure{B,K,V}, key::L) where {N,B<:Box{N},K,V,L<:Union{<:CartesianIndex{N},<:SVNT{N}}} = get(boxmeas.vals, key, zero(V))
Base.setindex!(boxmeas::BoxMeasure{B}, val, key::Vararg{<:Integer,N}) where {N,B<:Box{N}} = setindex!(boxmeas.vals, val, key)
Base.setindex!(boxmeas::BoxMeasure{B}, val, key::L) where {N,B<:Box{N},L<:Union{<:CartesianIndex{N},<:SVNT{N}}} = setindex!(boxmeas.vals, val, key)
Base.fill!(boxmeas::BoxMeasure, val) = (for key in keys(boxmeas); boxmeas[key] = val; end; boxmeas)
Base.sizehint!(boxmeas::BoxMeasure, sz) = sizehint!(boxmeas.vals, sz)
Base.copy(boxmeas::BoxMeasure) = BoxMeasure(boxmeas.partition, copy(boxmeas.vals))
Base.deepcopy(boxmeas::BoxMeasure) = BoxMeasure(boxmeas.partition, deepcopy(boxmeas.vals))
SparseArrays.findnz(boxmeas::BoxMeasure) = (collect(keys(boxmeas)), collect(values(boxmeas)))

"""
    marginal(μ::BoxMeasure{Box{N}}; dim) -> BoxMeasure{Box{N-1}}

Compute the marginal distribution of μ along an axis given
by its dimension `dim`. 
"""
function marginal(μ⁺::BoxMeasure; dim)
    support = marginal(BoxSet(μ⁺); dim=dim)
    μ = BoxMeasure(support, eltype(μ⁺))

    for key⁺ in keys(μ⁺)
        key = tuple_deleteat(key⁺, dim)
        μ[key] += μ⁺[key⁺]
    end

    return μ
end

Core.@doc raw"""
    density(μ::BoxMeasure) -> Function

Return the measure `μ` as a callable density `g`, i.e.
```math
\int f(x) \, d\mu (x) = \int f(x)g(x) \, dx . 
```
"""
function density(μ::BoxMeasure)
    P = μ.partition
    function eval_density(x)
        xi = point_to_key(P, x)
        b = key_to_box(P, xi)
        return μ[xi] / volume(b)
    end
end

function Base.isapprox(
        l::BoxMeasure{B,K,V}, r::BoxMeasure{R,J,W}; 
        atol=0, rtol=Base.rtoldefault(V,W,atol), kwargs...
    ) where {B,K,V,R,J,W}
    
    l === r && return true
    atol_used = max(atol, rtol * max(norm(values(l)), norm(values(r))))
    for key in (keys(l) ∪ keys(r))
        isapprox(l[key], r[key]; atol=atol_used, rtol=rtol, kwargs...) || return false
    end
    return true
end

function Base.:(==)(l::BoxMeasure{B,K,V}, r::BoxMeasure{R,J,W}) where {B,K,V,R,J,W}
    l === r && return true
    for key in (keys(l) ∪ keys(r))
        l[key] == r[key] || return false
    end
    return true
end

gen_type(d::AbstractDict{K,V}, f) where {K,V} = Dict{K,(typeof ∘ f ∘ first ∘ values)(d)}
gen_type(d::OrderedDict{K,V}, f) where {K,V} = OrderedDict{K,(typeof ∘ f ∘ first ∘ values)(d)}

"""
    ∘(f, boxmeas::BoxMeasure) -> BoxMeasure
    ∘(boxmeas::BoxMeasure, F::BoxMap) -> BoxMeasure

Postcompose the function `f` with the `boxmeas`,
or precompose a BoxMap `F` with the `boxmeas` 
(by applying the Koopman operator). Note that 
the support of `BoxMeasure` must be forward-invariant
under `F`. 
"""
function ∘(f, boxmeas::BoxMeasure)
    D = gen_type(boxmeas.vals, f)
    BoxMeasure(
        boxmeas.partition, 
        D(key => f(val) for (key,val) in boxmeas.vals)
    )
end

function ∘(boxmeas::BoxMeasure, F::BoxMap)
    T = TransferOperator(F, BoxSet(boxmeas))
    return T'boxmeas
end

Base.:(*)(a::Number, boxmeas::BoxMeasure) = (x -> x*a) ∘ boxmeas
Base.:(*)(boxmeas::BoxMeasure, a::Number) = (x -> x*a) ∘ boxmeas
Base.:(/)(boxmeas::BoxMeasure, a::Number) = (x -> x/a) ∘ boxmeas
Base.:(-)(b::BoxMeasure) = -1 * b
Base.:(-)(b1::BoxMeasure, b2::BoxMeasure) = b1 + (-b2)

function Base.:(+)(b1::BoxMeasure, b2::BoxMeasure)
    b1.partition == b2.partition || throw(DomainError("Partitions of BoxMeasures do not match."))

    v1 = first(values(b1))
    D = gen_type(b2.vals, x -> x + v1)
    b = BoxMeasure(b1.partition, D())

    sizehint!(b, max(length(b1), length(b2)))
    for key in (keys(b1) ∪ keys(b2))
        b[key] = b1[key] + b2[key]
    end

    return b
end
