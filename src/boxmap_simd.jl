const SVNT{N,T} = Union{NTuple{N,T}, <:StaticVector{N,T}}

struct BoxMapCPUCache{simd,V,W}
    idx_base::SIMD.Vec{simd,Int}
    temp_points::V
    temp_points_vec::W
end

function Base.show(io::IO, g::SampledBoxMap{<:BoxMapCPUCache{simd}}) where {simd}
    center, radius = g.domain.center, g.domain.radius
    n = length(g.domain_points(center, radius)) * simd
    print(io, "BoxMap with $(n) sample points")
end

Base.iterate(c::BoxMapCPUCache) = (c.idx_base, Val(:temp_points))
Base.iterate(c::BoxMapCPUCache, ::Val{:temp_points}) = (c.temp_points, Val(:temp_points_vec))
Base.iterate(c::BoxMapCPUCache, ::Val{:temp_points_vec}) = (c.temp_points_vec, Val(:done))
Base.iterate(c::BoxMapCPUCache, ::Val{:done}) = nothing

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:cpu}) where {N,T}
    n, simd = length(points), Int(pick_vector_width(T))
    if n % simd != 0
        throw(DimensionMismatch("Number of test points $n is not divisible by $T SIMD capability $simd"))
    end
    gathered_points = tuple_vgather(points, simd)
    domain_points(center, radius) = gathered_points
    image_points(center, radius) = center
    idx_base = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    temp_points = Vector{T}(undef, N*simd*nthreads())
    temp_points_vec = reinterpret(SVector{N,T}, temp_points)
    return SampledBoxMap(map, domain, domain_points, image_points, BoxMapCPUCache(idx_base, temp_points, temp_points_vec))
end

function sample_adaptive(Df, center::SVector{N,T}, ::Val{simd}) where {N,T,simd} 
    D = Df(center)
    _, σ, Vt = svd(D)
    n = ceil.(Int, σ)
    d = argmax(n)
    n[d] = ceil(Int, n[d] / simd)
    h = 2.0 ./ (n .- 1.0)
    points = Array{SVector{N,SIMD.Vec{simd,T}}}(undef, n...)
    d = n[d]
    n[d] = n[d] * simd
    inds = CartesianIndices(tuple(n...))
    n[d] = d
    for i in 0 : prod(n) - 1
        points[i+1] = ntuple(Val(N)) do j
            SIMD.Vec{simd,T}(ntuple(Val(simd)) do k
                    m = getindex(inds[simd*i+k], j)
                    T(isone(n[j]) ? 0.0 : (m-1) * h[j] - 1.0)
                end
            )
        end
        points[i+1] = Vt'*points[i+1]
    end   
    @debug points
    return points 
end

function AdaptiveBoxMap(f, domain::Box{N,T}, accel::Val{:cpu}) where {N,T}
    Df = x -> ForwardDiff.jacobian(f, x)
    simd = Int(pick_vector_width(T))
    domain_points(center, radius) = sample_adaptive(Df, center, Val(simd))

    vertices = Array{SVector{N,T}}(undef, ntuple(k->2, N))
    for i in CartesianIndices(vertices)
        vertices[i] = ntuple(k -> (-1.0)^i[k], N)
    end
    # calculates the vertices of each box
    image_points(center, radius) = vertices

    idx_base = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    temp_points = Vector{T}(undef, N*simd*nthreads())
    temp_points_vec = reinterpret(SVector{N,T}, temp_points)
    
    return SampledBoxMap(f, domain, domain_points, image_points, BoxMapCPUCache(idx_base, temp_points, temp_points_vec))
end

@inbounds function map_boxes(g::SampledBoxMap{<:BoxMapCPUCache{simd},N}, source::BoxSet) where {simd,N}
    P, keys = source.partition, collect(source.set)
    image = [ Set{eltype(keys)}() for _ in 1:nthreads() ]
    idx_base, temp_vec, temp_points = g.acceleration
    @threads for key in keys
        tid  = (threadid() - 1) * simd
        idx  = idx_base + tid * N
        mapped_points = @view temp_points[tid+1:tid+simd]
        box  = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = g.map(@muladd p .* r .+ c)
            tuple_vscatter!(temp_vec, fp, idx)
            for q in mapped_points
                hit = point_to_key(P, q)
                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end

@inbounds function TransferOperator(g::SampledBoxMap{<:BoxMapCPUCache{simd},N}, source::BoxSet{<:BoxPartition}) where {simd,N}
    P = source.partition
    edges = [ Dict{Tuple{Int64,Int64},Float64}() for k = 1:nthreads() ]
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
        n = length(points) * simd
        for p in points
            fp = g.map(@muladd p .* r .+ c)
            tuple_vscatter!(temp_vec, fp, idx)
            for q in mapped_points
                hit = point_to_key(P, q)
                if !isnothing(hit) && hit in source.set
                    j = key_to_index[hit]
                    e = (i,j)
                    t_edges[e] = get(t_edges, e, 0.0) + 1.0/n
                end
            end
        end
    end
    return TransferOperator(boxlist, merge(edges...))
end

function tuple_vgather(
        v::V, idx::SIMD.Vec{simd,Int}# = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    ) where {N,T,simd,V<:AbstractArray{<:SVNT{N,T}}}

    vr = reinterpret(T,v)
    vo = ntuple(i -> vr[idx + i], Val(N))
    return vo
end

@inline function tuple_vgather(
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

# helper + compatibility functions
@inline function tuple_vgather_lazy(
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

function tuple_vscatter!(
        vo::VO, vi::VI, idx::SIMD.Vec{simd,I}
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:SVNT{N,SIMD.Vec{simd,T}},I<:Integer}
    
    for i in 1:N
        vo[idx + i] = vi[i]
    end
end

function tuple_vscatter!(
        vo::VO, vi::VI
    ) where {N,T,simd,VO<:AbstractArray{T},VI<:AbstractArray{<:SVNT{N,SIMD.Vec{simd,T}}}}

    idx = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    for j in 1:length(vi)
        tuple_vscatter!( vo, vi[j], idx + (j-1)*N*simd )
    end
end
