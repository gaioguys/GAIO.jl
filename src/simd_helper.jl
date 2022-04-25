#function tuple_vgather(v::V, simd, create_only_one_point) where V<:AbstractVector{SVector{N,T}} where {N,T}
@propagate_inbounds function tuple_vgather(
        v::V, simd, create_only_one_point
    ) where V<:AbstractVector{SV} where SV<:Union{NTuple{N,T}, <:StaticVector{N,T}} where {N,T}

    vr = reinterpret(T, v)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), simd))
    vo = ntuple(i -> vr[idx + i], N)
    return vo
end

#function tuple_vgather(v::V, simd) where V<:AbstractVector{SVector{N,T}} where {N,T}
@propagate_inbounds function tuple_vgather(
        v::V, simd
    ) where V<:AbstractVector{SV} where SV<:Union{NTuple{N,T}, <:StaticVector{N,T}} where {N,T}

    vo = Vector(
        map(0:simd:length(v)-simd) do i
            tuple_vgather(
                view(v, i+1:i+simd),
                simd,
                true
            )
        end
    )
    return vo
end

@propagate_inbounds function tuple_vscatter!(
        vo::VO, vi::VI; start_ind=1, idx=SIMD.Vec{simd,Int}(ntuple(i -> N*(i-1), Val(simd)))
    ) where {VO<:AbstractVector{T}, VI<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}}} where {N,T,simd}
    
    if @generated
        return quote
            @nexprs( $N, i -> vo[idx + start_ind + i] = vi[i] )
        end
    else
        for i in 1:N
            vo[idx + start_ind + i] = vi[i]
        end
    end
end

function get_vector_width(
        ::V
    ) where V<:AbstractVector{SV} where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:SVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    simd
end

#= 
#function tuple_vscatter(vo::SVector{N,SIMD.Vec{simd,T}}) where {N,T,simd}
@inbounds function tuple_vscatter(
        vo::SV
    ) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}

    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{T}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vo[i]
    end
    v = reinterpret(NTuple{N,T}, vr)
    return v
end

#function tuple_vscatter(vo::NTuple{N,SIMD.Vec{simd,Bool}}) where {N,simd}
@inbounds function tuple_vscatter(
        vo::SV
    ) where SV<:Union{NTuple{N,SIMD.Vec{simd,Bool}}, <:StaticVector{N,SIMD.Vec{simd,Bool}}} where {N,simd} 

    vor = reinterpret.(SIMD.Vec{simd,UInt8}, vo)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{UInt8}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vor[i]
    end
    v = reinterpret(NTuple{N,Bool}, vr)
    return v
end
 =#
