struct BoxMapCPUCache{simd,V,W}
    idx_base::SIMD.Vec{simd,Int}
    temp_vec::V
    temp_points::W
end

function BoxMapCPUCache(N, T)
    simd = Int(pick_vector_width(T))
    idx_base = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    temp_vec = Vector{T}(undef, N*simd*nthreads())
    temp_points = reinterpret(SVector{N,T}, temp_vec)
    BoxMapCPUCache(idx_base, temp_vec, temp_points)
end

BoxMapCPUCache(::Box{N,T}) where {N,T} = BoxMapCPUCache(N, T)

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:cpu}) where {N,T}
    n, simd = length(points), Int(pick_vector_width(T))
    if n % simd != 0
        throw(DimensionMismatch("Number of test points $n is not divisible by $T SIMD capability $simd"))
    end
    gathered_points = tuple_vgather(points, simd)
    domain_points = rescale(gathered_points)
    image_points = center
    return SampledBoxMap(map, domain, domain_points, image_points, BoxMapCPUCache(domain))
end

function sample_adaptive(Df, center::SVector{N,T}, ::Val{simd}) where {N,T,simd} 
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ) 
    d = argmax(@view n[1:N])
    r = n[d] % simd
    r = r == 0 ? r : simd - r
    n = SVector{N,Int}([i == d ? n[i] + r : n[i] for i in 1:N])
    h = 2.0./(n.-1)
    points = Array{SVector{N,T}}(undef, n.data)
    for i in CartesianIndices(points)
        points[i] = ntuple(k -> n[k]==1 ? 0.0 : (i[k]-1)*h[k]-1.0, N)
        points[i] = Vt'*points[i]
    end
    @debug points
    points_gathered = tuple_vgather(vec(points), simd)
    return points_gathered
end

function AdaptiveBoxMap(f, domain::Box{N,T}, accel::Val{:cpu}) where {N,T}
    Df(x) = ForwardDiff.jacobian(f, x)
    simd = Int(pick_vector_width(T))
    domain_points(center, radius) = rescale(center, radius, sample_adaptive(Df, center, Val(simd)))
    image_points = vertices
    return SampledBoxMap(f, domain, domain_points, image_points, BoxMapCPUCache(domain))
end

@inbounds @muladd function map_boxes(g::SampledBoxMap{<:BoxMapCPUCache{simd},N}, source::BoxSet{B,Q,S}) where {simd,N,B,Q,S}
    P = source.partition
    idx_base, temp_vec, temp_points = g.acceleration
    @floop for box in source
        tid = (threadid() - 1) * simd
        idx = idx_base + tid * N
        mapped_points = @view temp_points[tid+1:tid+simd]
        c, r = box.center, box.radius
        for p in g.domain_points(c, r)
            fp = g.map(p)
            tuple_vscatter!(temp_vec, fp, idx)
            for q in mapped_points
                hitbox = point_to_box(P, q)
                isnothing(hitbox) && continue
                r = hitbox.radius
                for ip in g.image_points(q, r)
                    hit = point_to_key(P, ip)
                    isnothing(hit) && continue
                    @reduce(image = union!(S(), hit))
                end
            end
        end
    end
    return BoxSet(P, image)
end

@inbounds function TransferOperator(g::SampledBoxMap{<:BoxMapCPUCache{simd},N}, source::BoxSet{<:BoxPartition}) where {simd,N}
    P = source.partition
    edges = [ Dict{Tuple{Int64,Int64},Float64}() for _ in 1:nthreads() ]
    boxlist = BoxList(source)
    key_to_index = invert_vector(boxlist.keylist)
    idx_base, temp_vec, temp_points = g.acceleration
    @threads for i in 1:length(boxlist)#key in keys
        tid  = (threadid() - 1) * simd
        idx  = idx_base + tid * N
        mapped_points = @view temp_points[tid+1:tid+simd]
        t_edges = edges[threadid()]
        box  = key_to_box(P, boxlist[i])
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        inv_n = 1. / (length(points) * simd)
        for p in points
            fp = g.map(p)
            tuple_vscatter!(temp_vec, fp, idx)
            for q in mapped_points
                hit = point_to_key(P, q)
                if !isnothing(hit) && hit in source.set
                    j = key_to_index[hit]
                    e = (i,j)
                    t_edges[e] = get(t_edges, e, 0.) + inv_n
                end
            end
        end
    end
    return TransferOperator(boxlist, merge(edges...))
end

# helper + compatibility functions
function Base.show(io::IO, g::SampledBoxMap{C}) where {simd,C<:BoxMapCPUCache{simd}}
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius)) * simd
    print(io, "BoxMap with $(n) sample points")
end

Base.iterate(c::BoxMapCPUCache) = (c.idx_base, Val(:temp_vec))
Base.iterate(c::BoxMapCPUCache, ::Val{:temp_vec}) = (c.temp_vec, Val(:temp_points))
Base.iterate(c::BoxMapCPUCache, ::Val{:temp_points}) = (c.temp_points, Val(:done))
Base.iterate(c::BoxMapCPUCache, ::Val{:done}) = nothing

Base.@propagate_inbounds function tuple_vgather(
        v::V, idx::SIMD.Vec{simd,Int}# = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    ) where {N,T,simd,V<:AbstractArray{<:SVNT{N,T}}}

    vr = reinterpret(T,v)
    vo = ntuple(i -> vr[idx + i], Val(N))
    return vo
end

Base.@propagate_inbounds function tuple_vgather(
        v::V, simd::Integer
    ) where {N,T,V<:AbstractArray{<:SVNT{N,T}}}

    n = length(v)
    m = n ÷ simd
    @boundscheck if n != m * simd
        throw(DimensionMismatch("length of input ($n) % simd ($simd) != 0"))
    end
    vr = reinterpret(T, v)
    vo = Vector{SVector{N,SIMD.Vec{simd,T}}}(undef, m)#ntuple(i -> vr[idx + i], Val(N))
    idx = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    for i in 1:m
        vo[i] = ntuple( j -> vr[idx + (i-1)*N*simd + j], Val(N) )
    end
    return vo
end

Base.@propagate_inbounds function tuple_vgather_lazy(
        v::V, simd
    ) where {N,T,V<:AbstractArray{<:SVNT{N,T}}}
    
    n = length(v)
    m = n ÷ simd
    @boundscheck if n != m * simd
        throw(DimensionMismatch("length of input ($n) % simd ($simd) != 0"))
    end
    vr = v |>
        x -> reinterpret(T, v) |> 
        x -> reshape(x, (N,simd,m)) |> 
        x -> PermutedDimsArray(x, (2,1,3)) |> 
        x -> reshape(x, (N*n,)) |>
        x -> reinterpret(SVector{N,SIMD.Vec{simd,T}}, x)
    return vr
end

Base.@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI, idx::SIMD.Vec{simd,I}
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:SVNT{N,SIMD.Vec{simd,T}},I<:Integer}
    
    for i in 1:N
        vo[idx + i] = vi[i]
    end
end

Base.@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:AbstractArray{<:SVNT{N,SIMD.Vec{simd,T}}}}

    idx = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    for j in 1:length(vi)
        tuple_vscatter!( vo, vi[j], idx + (j-1)*N*simd )
    end
end
