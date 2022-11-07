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
struct BoxFun{B,W,S<:BoxSet{B},V<:Vector{W}} <: AbstractSparseVector{W,Int}
    support::S
    vals::V
end

# ensure that `BoxFun` uses an `OrderedSet`
function BoxFun(support::M, vals::V) where {B,P,S<:OrderedSet,M<:BoxSet{B,P,S},W,V<:AbstractSparseVector{W}}
    BoxFun{B,W,S,V}(support, vals)
end

function BoxFun(support::M, vals::V) where {B,P,S,M<:BoxSet{B,P,S},W,V<:AbstractSparseVector{W}}
    BoxFun{B,W,S,V}(BoxSet(support.partition, OrderedSet(support.set)), vals)
end

Base.length(fun::BoxFun) = length(fun.support)

function Base.show(io::IO, g::BoxFun)
    print(io, "BoxFun over $(g.support)")
end

Base.show(io::IO, ::MIME"text/plain", fun::BoxFun) = show(io, fun)

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
    sum(boxfun.support.set) do key
        box = key_to_box(boxfun.support.partition, key)
        i = getkeyindex(boxfun.support, key)
        val = boxfun.vals[i]
        volume(box) * f(val)
    end
end

function Base.sum(f, boxfun::BoxFun{B,W,S}, boxset::BoxSet{D,P,R}) where {B,D,W,I,R<:AbstractSet{I},P,S<:Boxset{<:B,<:P,<:AbstractSet{I}}}
    sum(boxfun.support.set ∩ boxset.set) do key
        box = key_to_box(boxfun.support.partition, key)
        i = getkeyindex(boxfun.support, key)
        val = boxfun.vals[i]
        volume(box) * f(val)
    end
end

function Base.sum(f, boxfun::BoxFun{B,W,S}, boxset::BoxSet{D,P,R}) where {B,W,S,D,P,R}
    support = boxfun.support.partition[boxset]
    sum(f, boxfun, support)
end

(boxfun::BoxFun)(boxset::BoxSet) = sum(identity, boxfun, boxset)

LinearAlgebra.norm(boxfun::BoxFun) = norm(boxfun.vals)

function LinearAlgebra.normalize!(boxfun::BoxFun)
    λ = inv(norm(boxfun))
    map!(x -> λ*x, boxfun.vals, boxfun.vals)
    return boxfun
end

function Base.getindex(boxfun::BoxFun{B,W}, key) where {B,W}
    i = getkeyindex(boxfun.support, key)
    isnothing(i) ? zero(W) : boxfun.vals[i]
end

import Base: ∘

"""
    ∘(f, boxfun::BoxFun) -> BoxFun

Compose the function `f` with the `boxfun`. 
"""
∘(f, boxfun::BoxFun{P,K,V}) where {P,K,V} = BoxFun(boxfun.support, map(f, boxfun.vals))

