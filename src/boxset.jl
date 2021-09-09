struct BoxSet{P <: AbstractBoxPartition,S <: AbstractSet}
    partition::P
    set::S
end

function BoxSet(domain::Box{N,T}, dims::NTuple{N,Int}) where {N,T}
    P = BoxPartition(domain, dims)
    return BoxSet(P, Set{keytype(typeof(P))}())
end

function BoxSet(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int64,N)...)
    BoxSet(domain, dims)
end

BoxSet(domain::Box{1,T}, dims::Int) where {T} = BoxPartition(domain, (dims,))

function Base.show(io::IO, boxset::BoxSet) 
    size = length(boxset.set)
    dim = length(boxset.partition.domain.center)
    print(io, "$size-element BoxSet in dimension $dim")
end

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

        if key !== nothing
            push!(set, key)
        end
    end

    return BoxSet(partition, set)
end

function Base.getindex(boxset::BoxSet, points_or_point)
    getindex(boxset.partition, points_or_point)
end


function Base.getindex(partition::AbstractBoxPartition, ::Colon)
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
Base.eltype(::Type{BoxSet{P,S}}) where {P <: AbstractBoxPartition{B},S} where B = B
Base.iterate(boxset::BoxSet, state...) = iterate((key_to_box(boxset.partition, key) for key in boxset.set), state...)

function subdivide(boxset::BoxSet{<:BoxPartition,S}, dim::Int) where {S}
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

function subdivide!(boxset::BoxSet{<:TreePartition}, key::Tuple{Int,Int})
    @assert key in boxset.set

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
