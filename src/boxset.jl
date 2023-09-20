"""
    BoxSet(partition, indices::AbstractSet)

Internal data structure to hold boxes within a partition. 

Constructors:
* set of all boxes in partition / box set `P`:
```julia
B = cover(P, :)    
```
* cover the point `x`, or points `x = [x_1, x_2, x_3] # etc ...` using boxes from `P`
```julia
B = cover(P, x)
```    
* a covering of `S` using boxes from `P`
```julia
S = [Box(center_1, radius_1), Box(center_2, radius_2), Box(center_3, radius_3)] # etc... 
B = cover(P, S)
```

Fields:
* `partition`:  the partition that the set is defined over
* `set`:        set of partition-keys corresponding to the boxes in the set

Most set operations such as 
```julia
union, intersect, setdiff, symdiff, issubset, isdisjoint, issetequal, isempty, length, # etc...
```
are supported. 

"""
struct BoxSet{B,P<:AbstractBoxPartition{B},S<:AbstractSet} <: AbstractSet{B}
    partition::P
    set::S
end

function BoxSet{B,P,S}(partition::P) where {B,P<:AbstractBoxPartition{B},S<:AbstractSet}
    return BoxSet{B,P,S}(partition, S())
end

function BoxSet(partition::P; settype=Set{keytype(P)}) where {B,P<:AbstractBoxPartition{B}}
    return BoxSet{B,P,settype}(partition)
end

function BoxSet(N::BoxSet, filters; settype=Set)
    BoxSet(N.partition, settype(key for (key,filt) in zip(N.set, filters) if filt))
end

Base.show(io::IO, boxset::BoxSet) = print(io, "$(length(boxset)) - element BoxSet in ", boxset.partition)

function Base.show(io::IO, m::MIME"text/plain", boxset::BoxSet)
    print(io, "$(length(boxset)) - element BoxSet in ")
    show(io, m, boxset.partition)
end

function Base.getindex(partition::P, key) where P<:AbstractBoxPartition
    #eltype(point) <: Number || return map(x->partition[x], key)
    return key_to_box(P, key)
end

Base.getindex(partition::P, ::Colon) where P<:AbstractBoxPartition = collect(keys(partition))
Base.getindex(partition::P, c::CartesianIndex) where P<:AbstractBoxPartition = partition[c.I]

"""
* `BoxSet` constructors:
    * set of all boxes in partition / box set `P`:
    ```julia
    B = cover(P, :)    
    ```
    * cover the point `x`, or points `x = [x_1, x_2, x_3] # etc ...` using boxes from `P`
    ```julia
    B = cover(P, x)
    ```    
    * a covering of `S` using boxes from `P`
    ```julia
    S = [Box(center_1, radius_1), Box(center_2, radius_2), Box(center_3, radius_3)] # etc... 
    B = cover(P, S)
    ```

Return a subset of the partition or box set `P` based on the second argument. 
"""
function cover end

iscoverable(::T) where {T<:Union{<:Box,<:IntervalBox,<:Nothing,<:Colon}} = true
iscoverable(::T) where {T<:Union{<:AbstractArray{<:Number},NTuple{<:Any,<:Number}}} = true
iscoverable(::T) where {T} = false

function cover(B::BoxSet, x)
    A = cover(B.partition, x)
    return B ∩ A
end

function cover(partition::P, x) where {P<:AbstractBoxPartition}
    S = BoxSet(partition, Set{keytype(P)}())
    for object in x
        iscoverable(object) || throw(MethodError(cover, (partition, object)))
        union!(S, cover(partition, object))
    end
    return S
end

function cover(partition::P, ::Colon) where {P<:AbstractBoxPartition}
    return BoxSet(partition, Set{keytype(P)}(keys(partition)))
end

cover(partition::P, int::IntervalBox) where {P<:AbstractBoxPartition} = cover(partition, Box(int))
cover(partition::P, ::Nothing) where {P<:AbstractBoxPartition} = BoxSet(partition, Set{keytype(P)}())
cover(partition::P, x::Number) where {T,P<:AbstractBoxPartition{<:Box{1,T}}} = cover(partition, (x,))

function cover(partition::P, point::T) where {P<:AbstractBoxPartition,N,W<:Number,T<:Union{<:AbstractArray{W},<:NTuple{N,W}}}
    key = point_to_key(partition, point)
    B = BoxSet(partition, Set{keytype(P)}())
    isnothing(key) || push!(B, key)
    return B
end

function cover(partition::P, box::Box{N,T}) where {N,T,P<:BoxPartition}
    c = box.center
    r = box.radius .- 10*eps(T)
    key_lo = bounded_point_to_key(partition, c .- r)
    key_hi = bounded_point_to_key(partition, c .+ r)
    carts = CartesianIndices(ntuple(i -> key_lo[i]:key_hi[i], Val(N)))
    return BoxSet(partition, Set{keytype(P)}(Tuple(c) for c in carts))
end

function cover(partition::P, box::Box{N,R}) where {N,R,T,I,P<:TreePartition{N,T,I}}
    K = keytype(P)
    keys = Set{K}()
    box = partition.domain ∩ box
    isnothing(box) && return keys

    queue = Tuple{I,K}[(1, K((1, ntuple(_->1,N))))]
    while !isempty(queue)
        node_idx, key = pop!(queue)
        node = partition.nodes[node_idx]

        if isleaf(node)
            keys = keys ⊔ key
        else
            c1_idx, c2_idx = node
            depth, cart = key
            
            dim = (depth - 1) % N + 1
            key1 = (depth+1, Base.setindex(cart, 2 * cart[dim] - 1, dim))
            key2 = (depth+1, Base.setindex(cart, 2 * cart[dim], dim))
            
            Q = BoxPartition(partition, depth+1)
            box1 = key_to_box(Q, key1[2])
            box2 = key_to_box(Q, key2[2])

            isnothing(box1 ∩ box) || push!(queue, (c1_idx, key1))
            isnothing(box2 ∩ box) || push!(queue, (c2_idx, key2))
        end
    end
    return BoxSet(partition, keys)
end

for op in (:union, :intersect, :setdiff, :symdiff)
    op! = Symbol(op, :!) 
    boundscheck = quote
        @boundscheck for i in 1:length(sets)-1
            if !(sets[i].partition == sets[i+1].partition)
                throw(DomainError((sets[i].partition, sets[i+1].partition), "Partitions of boxsets in operation do not match."))
            end
        end
    end
    @eval begin
        @inline function Base.$op(sets::BoxSet...)
            $boundscheck
            return BoxSet(sets[1].partition, $op((s.set for s in sets)...))
        end

        @inline function Base.$op!(sets::BoxSet...)
            $boundscheck
            $op!((s.set for s in sets)...)
            return sets[1]
        end
    end
end

for op in (:issubset, :isdisjoint, :issetequal, :(==))
    @eval Base.$op(b1::BoxSet, b2::BoxSet) = b1.partition == b2.partition && $op(b1.set, b2.set)
end

function max_radius(boxset::BoxSet{B,P,S}) where {N,T,I,B,P<:TreePartition{N,T,I},S}
    min_depth = minimum(depth for (depth, cart) in boxset.set)
    Q = BoxPartition(boxset.partition, min_depth)
    _, r = key_to_box(Q, ntuple(_->one(I), Val(N)))
    return r
end

max_radius(boxset::BoxSet{B,P,S}) where {B,P<:BoxPartition,S} = first(boxset).radius
Base.isempty(boxset::BoxSet) = isempty(boxset.set)
Base.empty!(boxset::BoxSet) = (empty!(boxset.set); boxset)
Base.copy(boxset::BoxSet) = BoxSet(boxset.partition, copy(boxset.set))
Base.length(boxset::BoxSet) = length(boxset.set)
Base.keys(boxset::BoxSet) = keys(boxset.set)
Base.push!(boxset::BoxSet, key) = (push!(boxset.set, key); boxset)
Base.sizehint!(boxset::BoxSet, size) = (sizehint!(boxset.set, size); boxset)
Base.eltype(::Type{<:BoxSet{B}}) where B = B
SplittablesBase.amount(boxset::BoxSet) = SplittablesBase.amount(boxset.set)

function Base.iterate(boxset::BoxSet, state...)
    itr = iterate(boxset.set, state...)
    isnothing(itr) && return itr
    (key, j) = itr
    box = #= @inbounds =# key_to_box(boxset.partition, key)
    return (box, j)
end

function SplittablesBase.halve(boxset::BoxSet)
    P = boxset.partition
    left, right = SplittablesBase.halve(boxset.set)
    liter = (
        #= @inbounds =# key_to_box(P, key)
        for key in left
    )
    riter = (
        #= @inbounds =# key_to_box(P, key)
        for key in right
    )
    return (liter, riter)
end

"""
    neighborhood(B::BoxSet) -> BoxSet
    nbhd(B::BoxSet) -> BoxSet

Return a one-box wide neighborhood of a BoxSet `B`. 
"""
function neighborhood(B::BoxSet)
    nbhd = cover( B.partition, (Box(c, 1.2 .* r) for (c, r) in B) )
    return setdiff!(nbhd, B)
end

function neighborhood(B::BoxSet{R,Q}) where {N,R,Q<:BoxPartition{N}}
    P = B.partition
    C = empty!(copy(B))

    surrounding = CartesianIndices(ntuple(_-> -1:1, N))
    function _nbhd(key)
        keygen = (key .+ Tuple(cartesian_ind) for cartesian_ind in surrounding)
        (x for x in keygen if checkbounds(Bool, P, x))
    end

    for key in B.set
        union!(C.set, _nbhd(key))
    end

    return setdiff!(C, B)
end

function subdivide(boxset::BoxSet{B,P,S}, dim) where {B,P<:BoxPartition,S}
    set = S()
    sizehint!(set, 2*length(boxset.set))

    for key in boxset.set
        child1 = Base.setindex(key, 2 * key[dim] - 1, dim)
        child2 = Base.setindex(key, 2 * key[dim], dim)

        push!(set, child1, child2)
    end

    return BoxSet(subdivide(boxset.partition, dim), set)
end

function subdivide!(boxset::BoxSet{B,P,S}, key::Tuple{J,NTuple{N,K}}) where {B,N,P<:TreePartition{N},S,J,K}
    key in boxset.set || throw(KeyError(key))

    depth, cart = key
    dim = (depth - 1) % N + 1

    delete!(boxset.set, key)

    tree = boxset.partition
    subdivide!(tree, key)

    child1 = (depth+1, Base.setindex(cart, 2 * cart[dim] - 1, dim))
    child2 = (depth+1, Base.setindex(cart, 2 * cart[dim], dim))

    push!(boxset.set, child1, child2)

    return boxset
end

"""
    subdivide(B::BoxSet{<:Any,<:Any,<:TreePartition}) -> BoxSet

Bisect every box in `boxset` along an axis, giving rise to a new 
partition of the domain, with double the amount of boxes. 
Axis along which to bisect depends on the depth of the nodes. 
"""
function subdivide(boxset::BoxSet{B,P,S}, dim=1) where {B,P<:TreePartition,S}
    boxset_new = BoxSet(copy(boxset.partition), copy(boxset.set))
    #boxset_new = BoxSet(boxset.partition, copy(boxset.set))
    sizehint!(boxset_new, 2*length(boxset_new))

    for key in boxset.set
        subdivide!(boxset_new, key)
    end

    return boxset_new
end
