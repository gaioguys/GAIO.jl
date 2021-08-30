abstract type BoxMap end

struct SampledBoxMap{F,P,I} <: BoxMap
    map::F       
    domain_points::P
    image_points::I    
end

function PointDiscretizedMap(map, points::AbstractArray) 
    domain_points = (center, radius) -> points
    image_points = (center, radius) -> center
    return SampledBoxMap(map, domain_points, image_points)
end

boxmap(f, points) = PointDiscretizedMap(f, points)

function Base.show(io::IO, g::PointDiscretizedMap)
    print(io, "PointDiscretizedMap with $(length(g.points)) sample points")

function sample_adaptive(Df, center::SVector{N,T}) where {N,T}
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ) 
    h = 2.0./(n.-1)
    points = Array{SVector{N,T}}(undef, ntuple(i->n[i], N))
    for i in CartesianIndices(points)
        points[i] = ntuple(k -> n[k]==1 ? 0.0 : (i[k]-1)*h[k]-1.0, N)
        points[i] = Vt'*points[i]
    end   
    @show points
    return points 
end

function AdaptiveBoxMap(f, domain::Box{N,T}) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    domain_points = (center, radius) -> sample_adaptive(Df, center)

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end   
    image_points = (center, radius) -> vertices
    return SampledBoxMap(f, domain_points, image_points)
end

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
    target_partition = iter.target_partition
    f = iter.boxmap

    j = 0
    while boxes_iter <= length(boxes) && j < length(workinput)
        source = boxes[boxes_iter]
        box = key_to_box(source_partition, source)
        center, radius = box.center, box.radius
        points = f.domain_points(center, radius)
        if j + length(points) > length(workinput)
            resize!(workinput, j+length(points))
        end
        for point in points
            j += 1
            workinput[j] = (center .+ radius.*point, source, box)
        end
        boxes_iter += 1
     end
    
    resize!(workoutput, j)
    
    if j > 0
        @Threads.threads for i = 1:length(workoutput)
            point, source, box = workinput[i]
            fp = f.map(point)
            target = point_to_key(target_partition, fp)
            workoutput[i] = target
        end
    end
    
    return boxes_iter
end

@inline function Base.iterate(iter::ParallelBoxIterator, state)
    boxes, boxes_iter, workinput, workoutput, workoutput_iter = state

    if workoutput_iter > length(workoutput)
        @show length(workoutput)
        boxes_iter = fill_cache!(iter, boxes, boxes_iter, workinput, workoutput, workoutput_iter)
        workoutput_iter = 1
    end

    if isempty(workoutput)
        return nothing
    else
        source = workinput[workoutput_iter][2]
        target = workoutput[workoutput_iter]
        workoutput_iter += 1
        state = (boxes, boxes_iter, workinput, workoutput, workoutput_iter)
        return (source, target), state
    end
end

@inline function Base.iterate(iter::ParallelBoxIterator)
    boxes = collect(iter.boxset.set)
    boxes_iter = 1
    box = key_to_box(iter.boxset.partition, boxes[boxes_iter])
 
    # tuple of (point, sourcekey, box)
    workinput_eltype = Tuple{typeof(box.center), sourcekeytype(typeof(iter)), eltype(iter.boxset)}
    workinput = Vector{workinput_eltype}(undef, 100)

    workoutput = Union{targetkeytype(typeof(iter)), Nothing}[]
    workoutput_iter = 1

    state = (boxes, boxes_iter, workinput, workoutput, workoutput_iter)
    return iterate(iter, state)
end

Base.length(iter::ParallelBoxIterator) = length(iter.boxmap.points) * length(iter.boxset)

function Base.eltype(t::Type{<:ParallelBoxIterator})
    return Union{Tuple{sourcekeytype(t),targetkeytype(t)},Tuple{sourcekeytype(t),Nothing}}
end

function map_boxes_new(g::BoxMap, source::BoxSet)
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for k = 1:nthreads() ]
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        for p in g.points
            fp = g.f(c.+r.*p)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    BoxSet(P, union(image...))
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

function (g::BoxMap)(source::BoxSet; target = nothing)
    if isnothing(target)
        return map_boxes(g, source)
    else
        return map_boxes_with_target(g, source, target)
    end
end

function invert_vector(x::AbstractVector{T}) where T
    dict = Dict{T,Int}()
    sizehint!(dict, length(x))

    for i in 1:length(x)
        dict[x[i]] = i
    end

    return dict
end


