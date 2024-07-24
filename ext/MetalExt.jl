module MetalExt
    
using GAIO, Metal, StaticArrays, MuladdMacro
using ThreadsX

#import Base.Iterators: Stateful, take
import Base: unsafe_trunc
import Base: @propagate_inbounds, @boundscheck
import Metal: Adapt
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, point_to_key, ⊔, SVNT

#export GPUSampledBoxMap

BoxMap(::Val{Symbol("GPUSampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(::Val{Symbol("gpusampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(accel::Val{:gpu}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)
    
struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

struct GPUSampledBoxMap{N,T,F<:Function} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::MtlArray{SVector{N,T}}
end

function GPUSampledBoxMap(map::F, domain::Box, n_samples::Integer) where {F}
    @assert n_samples ≤ 1024
    domain = mtl(domain)
    N = ndims(domain);  T = eltype(domain)

    rn = 2 .* Metal.rand(T, N * n_samples) .- 1
    domain_points = reinterpret( SVector{N,T}, rn )

    GPUSampledBoxMap{N,T,F}(map, domain, domain_points)
end

@muladd function map_boxes_kernel!(g, P::Q, in_keys, out_keys, domain_points, offset) where {N,T,B<:Box{N,T},Q<:AbstractBoxPartition{B}}
    x, y = thread_position_in_grid_2d()
    x += offset
    k = in_keys[x]
    c, r = key_to_box(P, k)
    p = domain_points[y]
    p = @. c + r * p
    p = typesafe_map(P, g, p)
    hit = point_to_key(P, p, Val(:gpu))
    out_keys[x,y] = isnothing(hit) ? out_of_bounds(P) : hit
    return nothing
end

function map_boxes(G::GPUSampledBoxMap{F}, source::BoxSet{B,_Q,S}) where {B,_Q,S,F}
    g = G.map;  domain_points = G.domain_points
    P = mtl(source.partition)
    Q = typeof(P)

    in_keys = mtl(collect(source.set))
    n_keys = length(in_keys)
    n_samples = length(domain_points)

    out_keys = mtl(Array{keytype(Q)}( undef, (n_keys,n_samples) ))

    n_keys_per_group = 1024 ÷ n_samples
    n_groups = n_keys ÷ n_keys_per_group

    leftover = n_keys % (n_groups * n_keys_per_group)

    #@info "kernel initialization" n_keys n_samples n_keys_per_group n_groups leftover

    @metal threads=(n_keys_per_group,n_samples) groups=n_groups map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, 0i32)

    if leftover > 0
        @metal threads=(leftover,n_samples) groups=1 map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, leftover*i32)
    end

    Metal.synchronize()

    cpu_keys = Array(out_keys)

    #@info "this step needs to be optimized clearly" cpu_keys[1:5]

    #keyset = Set{keytype(Q)}()
    #union!(keyset, cpu_keys)
    keyset = ThreadsX.Set(cpu_keys)
    
    #@info "Set conversion complete"
    
    image = BoxSet(P, keyset)
    delete!(image.set, out_of_bounds(P))

    return image
end

function construct_transfers(
        G::GPUSampledBoxMap{F}, domain::BoxSet{R,_Q,S}, codomain::BoxSet{U,_H,W}
    ) where {N,T,R<:Box{N,T},_Q,S,U,_H,W,F}

    g = G.map;  domain_points = G.domain_points
    n_samples = length(G.domain_points)

    P = mtl(domain.partition)
    P2 = mtl(codomain.partition)
    P == P2 || throw(DomainError((P, P2), "Partitions of domain and codomain do not match. For GPU acceleration, they must be equal."))

    in_cpu = collect(domain.set)
    in_keys = mtl(in_cpu)
    n_keys = length(in_keys)
    
    Q = typeof(P)
    out_keys = mtl(Array{keytype(Q)}( undef, (n_keys,n_samples) ))

    n_keys_per_group = 1024 ÷ n_samples
    n_groups = n_keys ÷ n_keys_per_group

    leftover = n_keys % (n_groups * n_keys_per_group)

    #@info "kernel initialization" n_keys n_samples n_keys_per_group n_groups leftover

    @metal threads=(n_keys_per_group,n_samples) groups=n_groups map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, 0i32)

    if leftover > 0
        @metal threads=(leftover,n_samples) groups=1 map_boxes_kernel!(g, P, in_keys, out_keys, domain_points, leftover*i32)
    end

    Metal.synchronize()

    out_cpu = Array(out_keys)
    
    oob = out_of_bounds(P)
    K = keytype(Q)
    mat = Dict{Tuple{K,K},cu_reduce(T)}()
    for cartesian_ind in CartesianIndices(out_cpu)
        i, j = cartesian_ind.I
        key = in_cpu[i]
        hit = out_cpu[i,j]
        hit == oob && continue
        hit in codomain.set || continue
        mat = mat ⊔ ((hit,key) => 1)
    end

    return mat
end

function typesafe_map(::Q, g, p) where {N,T,B<:Box{N,T},Q<:AbstractBoxPartition{B}}
    convert(SVector{N,T}, g(p))
end

# hotfix to avoid errors due to cuda device-side printing
@propagate_inbounds function point_to_key(partition::BoxPartition{N,T,I}, point, ::Val{:gpu}) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, partition, x_ints)
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(partition))
    end
    return x_ints
end

function out_of_bounds(::P) where {N,T,I,P<:BoxPartition{N,T,I}}
    K = cu_reduce(keytype(P))
    K(ntuple(_->0, Val(N)))
end

function out_of_bounds(::P) where {N,T,I,P<:TreePartition{N,T,I}}
    K = cu_reduce(keytype(P))
    K((0, ntuple(_->0, Val(N))))
end

function Adapt.adapt_storage(::A, (c,r)::Box{N,T}) where {N,T,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT = cu_reduce(T)
    Box{N,TT}(c,r)
end

function Adapt.adapt_structure(a::A, b::BoxPartition{N,T,I}) where {N,T,I,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT, II = cu_reduce(T), cu_reduce(I)
    Adapt.adapt_storage(a, 
        BoxPartition{N,TT,II}(
            Box{N,TT}(b.domain...),
            SVector{N,TT}(b.left),
            SVector{N,TT}(b.scale),
            SVector{N,II}(b.dims)
        )
    )
end

function Adapt.adapt_structure(a::A, b::TreePartition{N,T,I}) where {N,T,I,A<:Union{<:Metal.MtlArrayAdaptor,<:Metal.Adaptor}}
    TT = cu_reduce(T)
    Adapt.adapt_storage(a,
        TreePartition(
            Box{N,TT}(b.domain...),
            Adapt.adapt(a, b.nodes)
        )
    )
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F},M}}

    MtlArray{SVector{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,F<:Integer,V<:AbstractArray{<:NTuple{N,F},M}}

    MtlArray{NTuple{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::Metal.MtlArrayAdaptor, x::V
    ) where {N,M,H<:Integer,F<:Integer,V<:AbstractArray{<:Tuple{H,NTuple{N,F}},M}}

    MtlArray{Tuple{H,NTuple{N,cu_reduce(F)}},M}(x)
end

cu_reduce(::Type{I}) where {I<:Integer} = Int32
cu_reduce(::Type{Int16}) = Int16
cu_reduce(::Type{Int8}) = Int8
cu_reduce(::Type{F}) where {F<:AbstractFloat} = Float32
cu_reduce(::Type{Float16}) = Float16
cu_reduce(::Type{<:NTuple{N,T}}) where {N,T} = NTuple{N,cu_reduce(T)}

end # module
