struct BoxSet{P <: BoxPartition,S <: AbstractSet}
    partition::P
    set::S
end

function boxset_empty(partition::P) where P <: BoxPartition
    return BoxSet(partition, Set{keytype(P)}())
end

function boxset_full(partition::BoxPartition)
    return BoxSet(partition, Set(keys_all(partition)))
end

for op in (:union, :intersect, :setdiff)
    op! = Symbol(op, :!)
    @eval begin
        function Base.$op(a::BoxSet, b::BoxSet)
            # TODO: verify that a.partition == b.partition
            return BoxSet(a.partition, $op(a.set, b.set))
        end

        function Base.$op!(a::BoxSet, b::BoxSet)
            # TODO: verify that a.partition == b.partition
            $op!(a.set, b.set)
            return a
        end
    end
end

Base.isempty(boxset::BoxSet) = isempty(boxset.set)
Base.length(boxset::BoxSet) = length(boxset.set)
Base.eltype(::Type{BoxSet{P,S}}) where {P <: BoxPartition{B},S} where B = B
Base.iterate(boxset::BoxSet, state...) = iterate((boxset.partition[key] for key in boxset.set), state...)

function subdivide(boxset::BoxSet{P,S}) where {P,S}
    set = S()
    sizehint!(set, 2 * length(boxset.set))

    for index in boxset.set
        child1, child2 = subdivide_key(boxset.partition, index)
        push!(set, child1)
        push!(set, child2)
    end

    return BoxSet(subdivide(boxset.partition), set)
end

function subdivide!(boxset::BoxSet{<:TreePartition}, key::Tuple{Int,Int})
    @assert key in boxset.set

    delete!(boxset.set, key)

    subdivide!(boxset.partition, key)

    child1, child2 = subdivide_key(RegularPartition(boxset.partition.domain, key[1]), key[2])

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
