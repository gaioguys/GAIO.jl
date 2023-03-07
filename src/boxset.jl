"""
    BoxSet(partition, indices::AbstractSet)

Internal data structure to hold boxes within a partition. 

Constructors (all constructors work with box sets as well):
* For all boxes in a partition: 
```julia
partition[:]
```
* For one box containing a point `x`: 
```julia
partition[x]
```
* For a covering of an iterable `S = [Box(c_1, r_1), Box(c_2, r_r)] # etc...`: 
```julia
partition[S]
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

function BoxSet(partition::P) where P <: AbstractBoxPartition
    return BoxSet(partition, Set{keytype(P)}())
end

Base.show(io::IO, boxset::BoxSet) = print(io, "$(length(boxset)) - element BoxSet in ", boxset.partition)

function Base.show(io::IO, m::MIME"text/plain", boxset::BoxSet)
    print(io, "$(length(boxset)) - element BoxSet in ")
    show(io, m, boxset.partition)
end

function Base.getindex(partition::P, key) where P<:AbstractBoxPartition
    eltype(point) <: Number || return map(x->partition[x], key)
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
function cover(partition::P, points) where P<:AbstractBoxPartition
    eltype(points) <: Number && ndims(partition) > 1 && return cover(partition, (points,))
    eltype(points) <: Box && return cover_boxes(partition, points)
    gen = (key for key in (point_to_key(partition, point) for point in points) if !isnothing(key))
    return BoxSet(partition, Set{keytype(P)}(gen))
end

cover(partition::P, box::Box) where P<:AbstractBoxPartition = cover(partition, (box,))
cover(partition::P, int::IntervalBox) where P<:AbstractBoxPartition = cover(partition, Box(int))

function cover(partition::P, ::Colon) where {P<:AbstractBoxPartition}
    return BoxSet(partition, Set{keytype(P)}(keys(partition)))
end

function cover(B::BoxSet, points)
    A = getindex(B.partition, points)
    return B ∩ A
end

"""
    cover_boxes(partition::BoxPartition, boxes)

Return a covering of an iterator of `Box`es using `Box`es from `partition`. 
Only covers the part of `boxes` which lies within `partition.domain`. 
This is returned by `cover(partition, boxes)`. 
"""
function cover_boxes(partition::P, boxes) where {N,T,I,P<:BoxPartition{N,T,I}}
    K = keytype(P)
    keys = Set{K}()
    vertex_keys = Matrix{I}(undef, N, 2)
    for box_in in boxes
        (any(isnan, box_in.center) || any(isnan, box_in.radius)) && continue
        box = Box{N,T}(box_in.center .- eps(T), box_in.radius .- eps(T))
        vertex_keys[:, 1] .= vertex_keys[:, 2] .= bounded_point_to_key(partition, box.center)
        for point in vertices(box)
            ints = bounded_point_to_key(partition, point)
            vertex_keys[:, 1] .= min.(vertex_keys[:, 1], ints)
            vertex_keys[:, 2] .= max.(vertex_keys[:, 2], ints)
        end
        C = CartesianIndices(ntuple(i -> vertex_keys[i, 1] : vertex_keys[i, 2], Val(N)))
        union!(keys, (K(i.I) for i in C))
    end
    return BoxSet(partition, keys)
end

function cover_boxes(partition::P, box_in::Box) where {N,T,I,P<:TreePartition{N,T,I}}
    K = keytype(P)
    keys = Set{K}()
    box = partition.domain ∩ box_in
    isnothing(box) && return keys

    queue = Tuple{I,K}[(1, K((1, ntuple(_->1,N))))]
    while !isempty(queue)
        node_idx, key = pop!(queue)
        node = tree.nodes[node_idx]

        if isleaf(node)
            keys = keys ⊔ key
        else
            c1_idx, c2_idx = node
            depth, cart = key
            
            dim = (depth - 1) % N + 1
            key1 = (depth+1, Base.setindex(cart, 2 * cart[dim] - 1, dim))
            key2 = (depth+1, Base.setindex(cart, 2 * cart[dim], dim))
            
            Q = BoxPartition(partition, depth)
            box1 = key_to_box(Q, key1[2])
            box2 = key_to_box(Q, key2[2])

            isnothing(box1 ∩ box) || push!(queue, (c1_idx, key1))
            isnothing(box2 ∩ box) || push!(queue, (c2_idx, key2))
        end
    end
    return keys
end

function cover_boxes(partition::P, boxes) where {N,T,I,P<:TreePartition{N,T,I}}
    K = keytype(P)
    keys = Set{K}()
    for box in boxes
        union!(keys, cover_boxes(partition, box))
    end
    return keys
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

Base.isempty(boxset::BoxSet) = isempty(boxset.set)
Base.empty!(boxset::BoxSet) = (empty!(boxset.set); boxset)
Base.copy(boxset::BoxSet) = BoxSet(boxset.partition, copy(boxset.set))
Base.length(boxset::BoxSet) = length(boxset.set)
Base.push!(boxset::BoxSet, key) = push!(boxset.set, key)
Base.sizehint!(boxset::BoxSet, size) = sizehint!(boxset.set, size)
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
