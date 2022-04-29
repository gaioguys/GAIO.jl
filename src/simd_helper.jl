const SVNT{N,T} = Union{NTuple{N,T}, <:StaticVector{N,T}}
const AV{T} = AbstractArray{T}

#function tuple_vgather(v::V, simd, create_only_one_point) where V<:AbstractVector{SVector{N,T}} where {N,T}
@propagate_inbounds function tuple_vgather(
        v::V, simd, idx = SIMD.Vec(ntuple( i -> N*(i-1), simd ))
    ) where {N,T,V<:AV{<:SVNT{N,T}}}

    vr = reinterpret(T, v)
    vo = ntuple(i -> vr[idx + i], Val(N))
    return vo
end

@propagate_inbounds function tuple_vgather_lazy(v::V, simd) where {N,T,V<:AV{<:SVNT{N,T}}}
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

function get_vector_width(::V) where {N,T,simd,V<:AV{<:SVNT{N,SIMD.Vec{simd,T}}}}
    simd
end
