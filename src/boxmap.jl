"""
    BoxMap

An abstract base type for a map which can map boxes to boxes.
A BoxMap `g` should be usable as follows:
```
g(boxset)
```
This should output a subset (i.e. a BoxSet) of the BoxPartition associated with `boxset`.

We should also be able to specify a target set:
```
g(source_boxset, target=target_boxset)
```
Here, we should get a subset of `target_boxset`, instead of a subset of the BoxPartition of `source_boxset`.
"""
abstract type BoxMap end

"""
    PointDiscretizedMap{F,P} <: BoxMap

A special type of a `BoxMap` which discretizes some function of type `F`
by mapping some points stored in a data structure of type `P`.
The points in `points` are assumed to lie in the range of [-1, 1] in each dimension.
"""
struct PointDiscretizedMap{F,P} <: BoxMap
    f::F
    points::P
end

"""
    boxmap(f, points) -> PointDiscretizedMap

Create a PointDiscretizedMap using the given function `f` and points `points`.
"""
boxmap(f, points) = PointDiscretizedMap(f, points)

struct ParallelBoxIterator{M <: BoxMap,SP,SS,TP}
    boxmap::M
    boxset::BoxSet{SP,SS}
    target_partition::TP
end

sourcekeytype(::Type{ParallelBoxIterator{M,SP,SS,TP}}) where {M,SP,SS,TP} = keytype(SP)
targetkeytype(::Type{ParallelBoxIterator{M,SP,SS,TP}}) where {M,SP,SS,TP} = keytype(TP)

function ParallelBoxIterator(boxmap::BoxMap, boxset::BoxSet)
    return ParallelBoxIterator(boxmap, boxset, boxset.partition)
end

@noinline function fill_cache!(iter, boxes, boxes_iter, workinput, workoutput, workoutput_iter)
    source_partition = iter.boxset.partition

    j = 0

    while boxes_iter <= length(boxes) && j < length(workinput)
        source = boxes[boxes_iter]
        boxes_iter += 1
        box = key_to_box(source_partition, source)

        for point in iter.boxmap.points
            j += 1
            workinput[j] = (point, source, box)
        end
    end

    resize!(workoutput, j)

    if j == 0
        return boxes_iter
    end

    target_partition = iter.target_partition
    f = iter.boxmap.f

    @Threads.threads for i = 1:length(workoutput)
        point, source, box = workinput[i]
        center, radius = box.center, box.radius
        fp = f(center .+ radius .* point)
        target = point_to_key(target_partition, fp)

        workoutput[i] = target
    end

    return boxes_iter
end

@inline function Base.iterate(iter::ParallelBoxIterator, state)
    boxes, boxes_iter, workinput, workoutput, workoutput_iter = state

    if workoutput_iter > length(workoutput)
        boxes_iter = fill_cache!(iter, boxes, boxes_iter, workinput, workoutput, workoutput_iter)

        if isempty(workoutput)
            return nothing
        end

        workoutput_iter = 1
    end

    source = workinput[workoutput_iter][2]
    target = workoutput[workoutput_iter]

    workoutput_iter += 1

    return (source, target), (boxes, boxes_iter, workinput, workoutput, workoutput_iter)
end

@inline function Base.iterate(iter::ParallelBoxIterator)
    boxes = collect(iter.boxset.set)
    boxes_iter = 1

    # vector of (point, sourcekey, box)
    workinput_eltype = Tuple{eltype(iter.boxmap.points),sourcekeytype(typeof(iter)),eltype(iter.boxset)}
    workinput = Vector{workinput_eltype}(undef, 100 * length(iter.boxmap.points))

    workoutput = Union{targetkeytype(typeof(iter)),Nothing}[]
    workoutput_iter = 1

    return iterate(iter, (boxes, boxes_iter, workinput, workoutput, workoutput_iter))
end

Base.length(iter::ParallelBoxIterator) = length(iter.boxmap.points) * length(iter.boxset)

function Base.eltype(t::Type{<:ParallelBoxIterator})
    return Union{Tuple{sourcekeytype(t),targetkeytype(t)},Tuple{sourcekeytype(t),Nothing}}
end

"""
    map_boxes(g::BoxMap, source::BoxSet) -> BoxSet

Alias for `g(source)`.
"""
function map_boxes(g::BoxMap, source::BoxSet)
    result = boxset_empty(source.partition)

    for (_, hit) in ParallelBoxIterator(g, source)
        if hit !== nothing # check that point was inside domain
            push!(result.set, hit)
        end
    end

    return result
end

"""
    map_boxes_with_target(g::BoxMap, source::BoxSet, target::BoxSet) -> BoxSet

Alias for `g(source, target=target)`.
"""
function map_boxes_with_target(g::BoxMap, source::BoxSet, target::BoxSet)
    result = boxset_empty(target.partition)

    for (_, hit) in ParallelBoxIterator(g, source, target.partition)
        if hit !== nothing # check that point was inside domain
            if hit in target.set
                push!(result.set, hit)
            end
        end
    end

    return result
end

"""
    (g::PointDiscretizedMap)(source::BoxSet [; target::BoxSet]) -> BoxSet

Compute the image of the BoxSet `source` applied to the PointDiscretizedMap `g`.
`source` and `target` may refer to the same set.

The semantics are as follows:
we check each Box `source_box` from `source` separately, and map all points from `g.points` (cf. definition of PointDiscretizedMap)
affinely to `source_box`. Then, we compute the image of each point using `g.f`. The next action depends, if the `target` parameter is specified:

* If it is not given, or `target=nothing`, then we look for the Box in the BoxPartition associated with `source` (given, that we map inside of it) and add it to the output BoxSet.
* If it is given, then we check if a mapped point is contained in the `target` BoxSet. If so, the corresponding Box is added to the output BoxSet.
In the case that `source` and `target` share the same underlying `BoxPartition`, this can be seen as computing `intersect(g(source), target)` with less intermediate steps.

After we did this for each box from `source`, we return the output BoxSet.
"""
function (g::PointDiscretizedMap)(source::BoxSet; target = nothing)
    if isnothing(target)
        return map_boxes(g, source)
    else
        return map_boxes_with_target(g, source, target)
    end
end

"""
    map_boxes_to_edges(g::BoxMap, source::BoxSet)

Build a digraph from `g(source, target=source)`
which has the boxes from `source` as vertices and edges from box `a` to box `b`,
iff. `b` lies in the image of `a` under `g`.

It returns the results in the format `(edges, vertex_mapping)` where `vertex_mapping` gives a map from BoxPartition keys
to indices ranging from 0 to n-1 (n denoting the number of vertices), and `edges`
being a list of the edges which are encoded by these indices.
"""
function map_boxes_to_edges(g::BoxMap, source::BoxSet)
    K = keytype(typeof(source.partition))
    edges = Set{Tuple{K,K}}()

    for (src, hit) in ParallelBoxIterator(g, source)
        if hit !== nothing # check that point was inside domain
            if hit in source.set
                push!(edges, (src, hit))
            end
        end
    end

    vertex_to_key = K[]
    key_to_vertex = Dict{K,Int}()
    keyset = keys(key_to_vertex)
    edges_translated = Tuple{Int,Int}[]
    
    for (src, hit) in edges
        if !(src in keyset)
            push!(vertex_to_key, src)
            key_to_vertex[src] = length(vertex_to_key)
        end

        if !(hit in keyset)
            push!(vertex_to_key, hit)
            key_to_vertex[hit] = length(vertex_to_key)
        end

        push!(edges_translated, (key_to_vertex[src], key_to_vertex[hit]))
    end

    return edges_translated, vertex_to_key
end
