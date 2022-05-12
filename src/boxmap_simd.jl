struct BoxMapCPUCache{simd,V}
    idx_base::SIMD.Vec{simd,Int}
    temp_points::V
end

function PointDiscretizedMap(map, domain::Box{N,T}, points, ::Val{:cpu}) where {N,T}
    n, simd = length(points), Int(pick_vector_width(T))
    if n % simd != 0
        throw(DimensionMismatch("Number of test points $n is not divisible by $T SIMD capability $simd"))
    end
    gathered_points = copy(tuple_vgather_lazy(points, simd))
    domain_points(center, radius) = gathered_points
    image_points(center, radius) = center
    idx_base = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    temp_points = Vector{T}(undef, N*simd*nthreads())
    temp_points_vec = reinterpret(NTuple{simd,SVector{N,T}}, temp_points)
    tp() = (temp_points, temp_points_vec)
    return SampledBoxMap(map, domain, domain_points, image_points, BoxMapCPUCache(idx_base, tp))
end

@inbounds function map_boxes(g::SampledBoxMap{<:BoxMapCPUCache{simd},N}, source::BoxSet) where {simd,N}
    P, keys, m = source.partition, collect(source.set), g.acceleration
    image = [ Set{eltype(keys)}() for _ in  1:nthreads() ]
    points = g.domain_points(P.domain.center, P.domain.radius)
    temp_points, temp_points_vec = m.temp_points()
    @threads for key in keys
        idx  = m.idx_base + (threadid() - 1) * N * simd
        box  = key_to_box(P, key)
        c, r = box.center, box.radius
        for p in points
            fp = g.map(@muladd p .* r .+ c)
            tuple_vscatter!(temp_points, fp, idx)
            for q in temp_points_vec[threadid()]
                hit = point_to_key(P, q)
                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end

@propagate_inbounds function tuple_vgather(
        v::V, simd, idx = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    ) where {N,T,V<:AV{<:SVNT{N,T}}}

    vr = reinterpret(T, v)
    vo = ntuple(i -> vr[idx + i], Val(N))
    return vo
end

@propagate_inbounds function tuple_vgather_lazy(
        v::V, simd
    ) where {N,T,V<:AV{<:SVNT{N,T}}}
    
    n = length(v)
    m = n ÷ simd
    @boundscheck if n - m * simd != 0
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

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI, idx::SIMD.Vec{simd,I}
    ) where {N,T,simd,VO<:AV{T},VI<:SVNT{N,SIMD.Vec{simd,T}},I<:Integer}
    
    if @generated
        return quote
            @nexprs( $N, i -> vo[idx + i] = vi[i] )
            return 
        end
    else
        for i in 1:N
            vo[idx + i] = vi[i]
        end
    end
end

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI
    ) where {N,T,simd,VO<:AV{T},VI<:SVNT{N,SIMD.Vec{simd,T}}}

    idx = SIMD.Vec{simd,Int}(ntuple( i -> N*(i-1), Val(simd) ))
    for j in 1:length(vi)
        tuple_vscatter!( vo, vi[j], idx + (j-1)*N*simd )
    end
end
