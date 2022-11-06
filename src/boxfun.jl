# TODO: cleanup type params. key type of partition must equal K
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

Base.length(fun::BoxFun) = length(fun.support)

function Base.show(io::IO, g::BoxFun)
    print(io, "BoxFun over with $(length(g.vals)) stored entries $(g.partition)")
end

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
    sum((key,val) -> (key in support.set) ? volume(box)*f(val) : zero(V), boxfun.vals)
end

(boxfun::BoxFun)(boxset::Union{Box,BoxSet}) = sum(identity, boxfun, boxset)

LinearAlgebra.norm(boxfun::BoxFun) = sqrt(sum(abs2, boxfun))

function LinearAlgebra.normalize!(boxfun::BoxFun)
    λ = inv(norm(boxfun))
    map!(x -> λ*x, boxfun.vals)
    return boxfun
end

Base.getindex(boxfun::BoxFun{B,K,V}, key) where {B,K,V} = get(boxfun.vals, key, zero(V))

import Base: ∘

"""
    ∘(f, boxfun::BoxFun) -> BoxFun

Compose the function `f` with the `boxfun`. 
"""
∘(f, boxfun::BoxFun{P,K,V}) where {P,K,V} = BoxFun(boxfun.support, map(f, boxfun.vals))
