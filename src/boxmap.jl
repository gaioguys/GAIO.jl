abstract type BoxMap end

struct PointDiscretizedMap{F,P} <: BoxMap
    f::F
    points::P
end

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

function map_boxes(g::BoxMap, source::BoxSet)
    result = boxset_empty(source.partition)

    for (_, hit) in ParallelBoxIterator(g, source)
        if hit !== nothing # check that point was inside domain
            push!(result.set, hit)
        end
    end

    return result
end

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

function (g::PointDiscretizedMap)(source::BoxSet; target = nothing)
    if isnothing(target)
        return map_boxes(g, source)
    else
        return map_boxes_with_target(g, source, target)
    end
end

function transition_graph(g::BoxMap, boxset::BoxSet)
    K = keytype(typeof(boxset.partition))
    edges = Dict{Tuple{K,K},Int}()

    for (src, hit) in ParallelBoxIterator(g, boxset)
        if hit !== nothing # check that point was inside domain
            if hit in boxset.set
                # TODO: this calculates the hash of (src, hit) twice
                # improve this once https://github.com/JuliaLang/julia/issues/31199 is resolved
                edges[(src, hit)] = get(edges, (src, hit), 0) + 1
            end
        end
    end

    return BoxGraph(boxset.partition, edges)
end
