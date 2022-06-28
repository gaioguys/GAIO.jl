"""
Internal data structure to hold sets

`partition`:  the partition that the set is defined over

`set`:        set of partition-keys corresponding to the boxes in the set

"""
struct BoxSet{P <: AbstractBoxPartition,S <: AbstractSet}
    partition::P
    set::S
end

function Base.show(io::IO, boxset::BoxSet) 
    size = length(boxset.set)
    print(io, "$size-element BoxSet in ", boxset.partition)
end

# TODO: replace with BoxSet(partition)
function boxset_empty(partition::P) where P <: AbstractBoxPartition
    return BoxSet(partition, Set{keytype(P)}())
end

function Base.getindex(partition::AbstractBoxPartition, points_or_point)
    # check if points is only a single point
    if eltype(points_or_point) <: Number
        points = (points_or_point,)
    else
        points = points_or_point
    end

    set = Set{keytype(typeof(partition))}()
    sizehint!(set, length(points))

    for point in points
        key = point_to_key(partition, point)

        if !isnothing(key)
            push!(set, key)
        end
    end

    return BoxSet(partition, set)
end

function Base.getindex(partition::AbstractBoxPartition, ::Colon)
    return BoxSet(partition, Set(keys(partition)))
end

for op in (:union, :intersect, :setdiff)
    op! = Symbol(op, :!) 
    boundscheck = quote
        @boundscheck for i in 1:length(sets)-1
            if !(sets[i].partition == sets[i+1].partition)
                throw(DomainError((sets[i].partition, sets[i+1].partition), "Partitions of boxsets in union do not match."))
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

Base.isempty(boxset::BoxSet) = isempty(boxset.set)
Base.length(boxset::BoxSet) = length(boxset.set)
Base.push!(boxset::BoxSet, key) = push!(boxset.set, key)
Base.sizehint!(boxset::BoxSet, size) = sizehint!(boxset.set, size)
Base.eltype(::Type{BoxSet{P,S}}) where {P <: AbstractBoxPartition{B},S} where B = B
Base.iterate(boxset::BoxSet, state...) = iterate((key_to_box(boxset.partition, key) for key in boxset.set), state...)

function subdivide(boxset::BoxSet{<:BoxPartition,S}, dim::Integer) where {S}
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

function subdivide!(boxset::BoxSet{<:TreePartition}, key::NTuple{2,<:Integer})
    !( key in boxset.set ) && throw(KeyError(key))

    delete!(boxset.set, key)

    tree = boxset.partition
    subdivide!(tree, key)

    child1, child2 = subdivide(BoxSet(tree.regular_partitions[key[1] + 1], Set(key[2]))).set

    push!(boxset.set, (key[1] + 1, child1))
    push!(boxset.set, (key[1] + 1, child2))

    return boxset
end

function subdivide(boxset::BoxSet{<:TreePartition})
    boxset_new = BoxSet(copy(boxset.partition), copy(boxset.set))

    for key in boxset.set
        subdivide!(boxset_new, key)
    end

    return boxset_new
end
