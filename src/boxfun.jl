# TODO: cleanup type params. key type of partition must equal K
struct BoxFun{P<:AbstractBoxPartition,K,V}
    partition::P
    dict::Dict{K,V}
end

Base.length(fun::BoxFun) = length(fun.dict)

function Base.show(io::IO, g::BoxFun)
    print(io, "BoxFun over $(g.partition)")
end

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

function ∘(f, boxfun::BoxFun{P,K,V}) where {P,K,V}
    fV = typeof(f(first(values(boxfun.dict))))
    dict = Dict{K,fV}()
    sizehint!(dict, length(boxfun.dict))

    for (key, value) in boxfun.dict
        dict[key] = f(value)
    end

    return BoxFun(boxfun.partition, dict)
end

