# helper function to access `Set` / `OrderedSet` internals
getkeyindex(dict::Dict, i) = (j = Base.ht_keyindex(dict, i); j > 0 ? j : nothing)
getkeyindex(set::Set, i) = (j = Base.ht_keyindex(set.dict, i); j > 0 ? j : nothing)
getkeyindex(dict::OrderedDict, i) = (j = OrderedCollections.ht_keyindex(dict, i, true); j > 0 ? j : nothing)
getkeyindex(set::OrderedSet, i) = (j = OrderedCollections.ht_keyindex(set.dict, i, true); j > 0 ? j : nothing)
getkeyindex(boxset::BoxSet, i) = getkeyindex(boxset.set, i)

"""
    BoxFun(partition, vals)

Discretization of a function over the domain `partition.domain`,
as a piecewise constant function over the boxes of `partition`. 
    
Implemented as a sparse vector over the indices of `partition`. 

Fields:
* `support`: `BoxSet{<:Any,<:Any,<:OrderedSet}` which covers all 
nonzero values of `vals`. 
* `vals`:    vector containing box values. 
* `vals`: A sparse vector whose indices are the box indices from 
`partition`, and whose values represent the values of the function. 

Methods implemented:

    length, LinearAlgebra.norm, LinearAlgebra.normalize!

Implementation detail:

The ith index of `vals` corresponds to the ith element of `support`, 
i.e. `vals[i]` corresponds to `support.set.dict.keys[i]`. This is 
a bit complicated, but is due to the lack of a direct constructor 
for `OrderedSet`s. To change this, add a comment to the PR 
`https://github.com/JuliaCollections/OrderedCollections.jl/pull/92`! 
This will hopefully bring attention to merge the PR. 

.
"""
struct BoxFun{B,K,V,P<:AbstractBoxPartition{B},D<:AbstractDict{K,V}} <: AbstractVector{V}
    partition::P
    vals::D
end

function Base.show(io::IO, g::BoxFun)
    print(io, "BoxFun in $(g.partition) with $(length(g.vals)) stored entries")
end

Base.length(fun::BoxFun) = length(fun.vals)
Base.keytype(::BoxFun{B,K,V}) where {B,K,V} = K
Base.eltype(::BoxFun{B,K,V}) where {B,K,V} = V
Base.values(fun::BoxFun) = values(fun.vals)
Base.show(io::IO, ::MIME"text/plain", fun::BoxFun) = show(io, fun)

function Base.iterate(boxfun::BoxFun{B,K,V}, i...) where {B,K,V}
    itr = iterate(boxfun.vals, i...)
    isnothing(itr) && return itr
    ((key, val), j) = itr
    (Pair{B,V}(key_to_box(boxfun.partition, key), val), j)
end


Core.@doc raw"""
    sum(f, boxfun::BoxFun)

Integrate a function `f` using `boxfun` as a density, that is,
if `boxfun` is the discretization of a measure ``\mu`` over the domain 
``Q``, then approximate the value of 
```math
\int_Q f \, d\mu .
```
"""
function Base.sum(f, boxfun::BoxFun, boxset=nothing)
    sum((box,val) -> volume(box)*f(val), boxfun)
end

function Base.sum(f, boxfun::BoxFun{B,K,V}, boxset::Union{Box,BoxSet}) where {B,K,V}
    support = boxfun.partition[boxset]
    sum( 
        volume(key_to_box(boxfun.partition, key)) * f(val) 
        for (key,val) in boxfun.dict if key in support.set 
    )
end

(boxfun::BoxFun)(boxset::Union{Box,BoxSet}) = sum(identity, boxfun, boxset)

LinearAlgebra.norm(boxfun::BoxFun) = sqrt(sum(abs2, boxfun))

function LinearAlgebra.normalize!(boxfun::BoxFun)
    λ = inv(norm(boxfun))
    map!(x -> λ*x, values(boxfun.vals))
    return boxfun
end

Base.:(==)(b1::BoxFun, b2::BoxFun) = b1.vals == b2.vals
Base.getindex(boxfun::BoxFun{B,K,V}, key) where {B,K,V} = get(boxfun.vals, key, zero(V))
Base.setindex!(boxfun::BoxFun, val, key) = setindex!(boxfun.vals, val, key)

function Base.isapprox(
        l::BoxFun{B,K,V}, r::BoxFun{R,J,W}; 
        atol=0, rtol=Base.rtoldefault(V,W,atol), kwargs...
    ) where {B,K,V,R,J,W}
    
    l === r && return true
    atol_used = max(atol, rtol * max(norm(values(l)), norm(values(r))))
    for key in (keys(l.vals) ∪ keys(r.vals))
        isapprox(l[key], r[key]; atol=atol_used, rtol=rtol, kwargs...) || return false
    end
    return true
end

import Base: ∘

"""
    ∘(f, boxfun::BoxFun) -> BoxFun

Compose the function `f` with the `boxfun`. 
"""
∘(f, boxfun::BoxFun{B,K,V,P,D}) where {B,K,V,P,D} = BoxFun(boxfun.partition, D(key => f(val) for (key,val) in boxfun.vals))
