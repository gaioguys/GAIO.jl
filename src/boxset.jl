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
    print(io, "$size-element BoxSet in ", boxset.partition)
end

Base.show(io::IO, ::MIME"text/plain", boxset::BoxSet) = show(io, boxset)

# TODO: replace with BoxSet(partition)
function boxset_empty(partition::P) where P <: AbstractBoxPartition
    return BoxSet(partition, Set{keytype(P)}())
end

function Base.getindex(partition::AbstractBoxPartition, points)
    eltype(points) <: Number && ndims(partition) > 1 && return partition[(points,)]
    gen = Iterators.filter(!isnothing, (point_to_key(partition, point) for point in points))
    return BoxSet(partition, Set(gen))
end

function Base.getindex(partition::AbstractBoxPartition, key::Integer)
    BoxSet(partition, Set([key]))
end

function Base.getindex(B::BoxSet, points)
    A = getindex(B.partition, points)
    return B ∩ A
end


function Base.getindex(partition::AbstractBoxPartition, ::Colon)
    return BoxSet(partition, Set(keys(partition)))
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

function subdivide(boxset::BoxSet{B,P,S}, dim) where {B,P<:BoxPartition,S}
    partition = boxset.partition
    box_indices = CartesianIndices(size(partition))

    partition_subdivided = subdivide(boxset.partition, dim)
    linear_indices = LinearIndices(CartesianIndices(size(partition_subdivided)))

    set = S()
    sizehint!(set, 2*length(boxset.set))

    for key in boxset.set
        box_index = box_indices[key].I

        child1 = Base.setindex(box_index, 2*box_index[dim]-1, dim)
        child2 = Base.setindex(box_index, 2*box_index[dim], dim)

        push!(set, linear_indices[CartesianIndex(child1)])
        push!(set, linear_indices[CartesianIndex(child2)])
    end

    return BoxSet(partition_subdivided, set)
end

"""
    subdivide!(boxset::BoxSet{<:Any,<:Any,<:TreePartition}, key::NTuple{2,<:Integer})

Subdivide a `TreePartition` at the node `key`. Dimension along which 
the node is subdivided depends on the depth of the node. 
"""
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
