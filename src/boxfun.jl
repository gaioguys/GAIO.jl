# TODO: cleanup type params. key type of partition must equal K
"""
    BoxFun(partition, dict)

Discretization of a function over the domain `partition.domain`,
as a piecewise constant function over the boxes of `partition`. 
    
Implemented as a sparse vector over the indices of `partition`. 

Fields:
* `partition`: An `AbstractBoxPartition` whose indices are used 
for `vals`
* `vals`: A sparse vector whose indices are the box indices from 
`partition`, and whose values represent the values of the function. 

Methods implemented:

    length, LinearAlgebra.norm, LinearAlgebra.normalize!

.
"""
struct BoxFun{P<:AbstractBoxPartition,K,V}
    partition::P
    dict::Dict{K,V}
end

Base.length(fun::BoxFun) = length(fun.dict)

function Base.show(io::IO, g::BoxFun)
    print(io, "BoxFun over $(g.partition)")
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
function Base.sum(f, boxfun::BoxFun{K,V}) where {K,V}
    sum(boxfun.dict) do pair
        key, value = pair
        box = key_to_box(boxfun.partition, key)
        volume(box) * f(value)
    end
end

LinearAlgebra.norm(boxfun::BoxFun) = sqrt(sum(abs2, boxfun))

function LinearAlgebra.normalize!(boxfun::BoxFun)
    λ = inv(norm(boxfun))
    map!(x -> λ*x, values(boxfun.dict))
    return boxfun
end

import Base: ∘

"""
Compose the function `f` with the `boxfun`. 
"""
function ∘(f, boxfun::BoxFun{P,K,V}) where {P,K,V}
    fV = typeof(f(first(values(boxfun.dict))))
    dict = Dict{K,fV}()
    sizehint!(dict, length(boxfun.dict))

    for (key, value) in boxfun.dict
        dict[key] = f(value)
    end

    return BoxFun(boxfun.partition, dict)
end

