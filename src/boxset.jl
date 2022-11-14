"""
    BoxSet(partition, indices::AbstractSet)

Internal data structure to hold boxes within a partition. 

Constructors:
* For all boxes in a partition: 
```julia
partition[:]
```
* For one box containing a point `x`: 
```julia
partition[x]
```
* For a covering of a set `S = [Box(c_1, r_1), Box(c_2, r_r)] # etc...`: 
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

function Base.show(io::IO, boxset::BoxSet)
    size = length(boxset.set)
    print(io, "$size - element BoxSet in ", boxset.partition)
end

Base.show(io::IO, ::MIME"text/plain", boxset::BoxSet) = show(io, boxset)

# TODO: replace with BoxSet(partition)
function boxset_empty(partition::P) where P <: AbstractBoxPartition
    return BoxSet(partition, Set{keytype(P)}())
end

"""
* getindex constructors:
    * set of all boxes in `P`:
    ```julia
    B = P[:]    
    ```
    * cover the point `x`, or points `x = [x_1, x_2, x_3] # etc ...` using boxes from `P`
    ```julia
    B = P[x]
    ```    
    * a covering of `S` using boxes from `P`
    ```julia
    S = [Box(center_1, radius_1), Box(center_2, radius_2), Box(center_3, radius_3)] # etc... 
    B = P[S]    
    ```

Return a subset of the partition `P` based on the second argument. 
"""
function Base.getindex(partition::P, points) where P<:AbstractBoxPartition
    eltype(points) <: Number && ndims(partition) > 1 && return partition[(points,)]
    eltype(points) <: Box && return cover_boxes(partition, points)
    gen = (key for key in (point_to_key(partition, point) for point in points) if !isnothing(key))
    return BoxSet(partition, Set{keytype(P)}(gen))
end

Base.getindex(partition::P, box::Box) where P<:AbstractBoxPartition = getindex(partition, (box,))

"""
    cover_boxes(partition::BoxPartition, boxes)

Return a covering of an iterator of `Box`es using `Box`es from `partition`. 
Only covers the part of `boxes` which lies within `partition.domain`. 
"""
function cover_boxes(partition::P, boxes; input_set=Set{keytype(P)}()) where {N,T,I,P<:BoxPartition{N,T,I}}
    keys = input_set
    vertex_keys = Matrix{I}(undef, N, 2)
    for box_in in boxes
        box = Box{N,T}(box_in.center, box_in.radius .- 2*eps(T))
        vertex_keys[:, 1] .= vertex_keys[:, 2] .= bounded_point_to_cartesian(partition, box.center)
        for point in vertices(box)
            ints = bounded_point_to_cartesian(partition, point)
            vertex_keys[:, 1] .= min.(vertex_keys[:, 1], ints)
            vertex_keys[:, 2] .= max.(vertex_keys[:, 2], ints)
        end
        C = CartesianIndices(ntuple(i -> vertex_keys[i, 1] : vertex_keys[i, 2], Val(N)))
        for ind in C
            key = cartesian_to_key(partition, ind.I)
            isnothing(key) && @error "Indexing went wrong somehow. Please open an issue with an MWE" ind.I key
            union!(keys, key)
        end
    end
    return BoxSet(partition, keys)
end

function Base.getindex(partition::P, key::Integer) where {P<:AbstractBoxPartition}
    BoxSet(partition, Set{keytype(P)}([key]))
end

function Base.getindex(B::BoxSet, points)
    A = getindex(B.partition, points)
    return B ∩ A
end

function Base.getindex(partition::P, ::Colon) where {P<:AbstractBoxPartition}
    return BoxSet(partition, Set{keytype(P)}(keys(partition)))
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
Base.iterate(boxset::BoxSet, state...) = iterate((key_to_box(boxset.partition, key) for key in boxset.set), state...)
SplittablesBase.amount(boxset::BoxSet) = SplittablesBase.amount(boxset.set)

function SplittablesBase.halve(boxset::BoxSet)
    P = boxset.partition
    left, right = SplittablesBase.halve(boxset.set)
    ((key_to_box(P, key) for key in left), (key_to_box(P, key) for key in right))
end

function subdivide(boxset::BoxSet{B,P,S}, dim) where {B,P<:BoxPartition{<:Any,<:Any,<:Any,<:IndexCartesian},S}
    partition_subdivided = subdivide(boxset.partition, dim)

    set = S()
    sizehint!(set, 2*length(boxset.set))

    for key in boxset.set
        child1 = Base.setindex(key, 2 * key[dim] - 1, dim)
        child2 = Base.setindex(key, 2 * key[dim], dim)

        push!(set, child1)
        push!(set, child2)
    end

    return BoxSet(partition_subdivided, set)
end

function subdivide(boxset::BoxSet{B,P,S}, dim) where {B,P<:BoxPartition{<:Any,<:Any,<:Any,<:IndexLinear},S}
    partition = boxset.partition
    partition_subdivided = subdivide(partition, dim)
    C, J = CartesianIndices(partition), LinearIndices(partition_subdivided)

    set = S()
    sizehint!(set, 2*length(boxset.set))

    for key_linear in boxset.set
        key = C[key_linear].I
        child1 = Base.setindex(key, 2 * key[dim] - 1, dim)
        child2 = Base.setindex(key, 2 * key[dim], dim)

        push!(set, J[CartesianIndex(child1)])
        push!(set, J[CartesianIndex(child2)])
    end

    return BoxSet(partition_subdivided, set)
end

function subdivide!(boxset::BoxSet{B,P,S}, key::NTuple{2,<:Integer}) where {B,P<:TreePartition,S}
    !( key in boxset.set ) && throw(KeyError(key))

    delete!(boxset.set, key)

    tree = boxset.partition
    subdivide!(tree, key)

    child1, child2 = subdivide(BoxSet(tree.regular_partitions[key[1] + 1], Set(key[2]))).set

    push!(boxset.set, (key[1] + 1, child1))
    push!(boxset.set, (key[1] + 1, child2))

    return boxset
end

"""
    subdivide(B::BoxSet{<:Any,<:Any,<:TreePartition}) -> BoxSet

Bisect every box in `boxset` along an axis, giving rise to a new 
partition of the domain, with double the amount of boxes. 
Axis along which to bisect depends on the depth of the nodes. 
"""
function subdivide(boxset::BoxSet{B,P,S}) where {B,P<:TreePartition,S}
    boxset_new = BoxSet(copy(boxset.partition), copy(boxset.set))

    for key in boxset.set
        subdivide!(boxset_new, key)
    end

    return boxset_new
end
