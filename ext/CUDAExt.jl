module CUDAExt

using GAIO, CUDA, StaticArrays, MuladdMacro

import Base.Iterators: Stateful, take
import Base: unsafe_trunc
import Base: @propagate_inbounds
import CUDA: Adapt
import GAIO: BoxMap, PointDiscretizedBoxMap, GridBoxMap, MonteCarloBoxMap
import GAIO: typesafe_map, map_boxes, construct_transfers, point_to_key, ⊔, SVNT

#export GPUSampledBoxMap

BoxMap(::Val{Symbol("GPUSampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(::Val{Symbol("gpusampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(accel::Val{:gpu}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)
    
struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

"""
    BoxMap(:gpu, map, domain; n_points) -> GPUSampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `GPUSampledBoxMap` defined 
on `Box`es. 

Uses the GPU's acceleration capabilities. 

By default uses a grid of sample points. 


    BoxMap(:montecarlo, :gpu, boxmap_args...)
    BoxMap(:grid, :gpu, boxmap_args...)
    BoxMap(:pointdiscretized, :gpu, boxmap_args...)
    BoxMap(:sampled, :gpu, boxmap_args...)

Type representing a dicretization of a map using 
sample points, which are mapped on the gpu. This 
type performs orders of magnitude faster than 
standard `SampledBoxMap`s when point mapping is the 
bottleneck. 

!!! warning "`image_points` with `GPUSampledBoxMap`"
    `GPUSampledBoxMap` makes NO use of the `image_points` 
    field in `SampledBoxMap`s. 

Fields:
* `map`:              map that defines the dynamical system.
* `domain`:           domain of the map, `B`.
* `domain_points`:    the spread of test points to be mapped forward 
                      in intersection algorithms.
                      WARNING: this differs from 
                      SampledBoxMap.domain_points in that it is a vector 
                      of "global" test points within [-1, 1]ᴺ. 


Requires a CUDA-capable gpu. 
"""
struct GPUSampledBoxMap{N,T,F<:Function} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::CuArray{SVector{N,T}}

    function GPUSampledBoxMap{N,T,F}(map::F, domain::Box, domain_points::CuArray{SVector{N,T}}) where {N,T,F<:Function}
        new{N,T,F}(map, Box{N,T}(domain...), domain_points)
    end
end

function GPUSampledBoxMap(map::F, domain::Box, domain_points::CuArray{SVector{N,T}}) where {N,T,F<:Function}
    GPUSampledBoxMap{N,T,F}(map, domain, domain_points)
end

function GPUSampledBoxMap(boxmap::SampledBoxMap{N}) where {N}
    c, r = boxmap.domain
    points = boxmap.domain_points(c, r)
    map!(x -> (x .- c) ./ r, points, points)    # send all points to [-1, 1]ᴺ
    GPUSampledBoxMap(boxmap.map, boxmap.domain, cu(points))
end

function BoxMap(symb::Symbol, ::Val{:gpu}, args...; kwargs...)
    F = BoxMap(symb, args...; kwargs...)
    GPUSampledBoxMap(F)
end

function map_boxes_kernel!(g, P, domain_points, in_keys, out_keys)
    nk = length(in_keys)*i32
    np = length(domain_points)*i32
    ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x #- 1i32
    stride = gridDim().x * blockDim().x
    len = nk * np #- 1i32
    for i in ind : stride : len
        m, n = CartesianIndices((np, nk))[i].I
        key  = in_keys[n]
        box  = key_to_box(P, key)
        c, r = box
        p    = domain_points[m]
        p    = @muladd p .* r .+ c
        fp   = typesafe_map(P, g, p)
        hit  = @inbounds point_to_key(P, fp, Val(:gpu))
        out_keys[i] = isnothing(hit) ? out_of_bounds(P) : hit
    end
end

function map_boxes(
        G::GPUSampledBoxMap{N,T}, source::BoxSet{B,Q,S}
    ) where {N,T,B,Q<:BoxGrid,S}

    g = G.boxmap
    p = g.domain_points(g.domain...).iter
    np = length(p)
    keys = Stateful(source.set)
    P = source.partition
    K = cu_reduce(keytype(Q))
    image = S()
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_cpu = collect(K, take(keys, stride))
        in_keys = CuArray{K,1}(in_cpu)
        nk = length(in_cpu)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        union!(image, out_cpu)
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(image, out_of_bounds(P))
    return BoxSet(P, image)
end

function construct_transfers(
        G::GPUSampledBoxMap, domain::BoxSet{R,Q,S}, codomain::BoxSet{U,H,W}
    ) where {N,T,R<:Box{N,T},Q,S,U,H,W}

    g = G.boxmap
    p = g.domain_points(g.domain...).iter
    np = length(p)
    keys = Stateful(domain.set)
    P = domain.partition
    P2 = codomain.partition
    P == P2 || throw(DomainError((P, P2), "Partitions of domain and codomain do not match. For GPU acceleration, they must be equal."))
    K = cu_reduce(keytype(Q))
    D = Dict{Tuple{K,K},cu_reduce(T)}
    mat = D()
    codomain = BoxSet(P2, S())
    oob = out_of_bounds(P)
    while !isnothing(keys.nextvalstate)
        stride = min(
            length(keys),
            available_array_memory() ÷ (sizeof(K) * 10 * (N + 1) * np)
        )
        in_cpu = collect(K, take(keys, stride))
        in_keys = CuArray{K,1}(in_cpu)
        nk = length(in_cpu)
        out_keys = CuArray{K,1}(undef, nk * np)
        launch_kernel_then_sync!(
            nk * np, map_boxes_kernel!, 
            g.map, P, p, in_keys, out_keys
        )
        out_cpu = Array{K,1}(out_keys)
        C = CartesianIndices((np, nk))
        for i in 1:nk*np
            _, n = C[i].I
            key, hit = in_cpu[n], out_cpu[i]
            hit == oob && continue
            hit in codomain.set || continue
            mat = mat ⊔ ((hit,key) => 1)
        end
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    return mat
end

# helper + compatibility functions
function Base.show(io::IO, g::GPUSampledBoxMap)
    n = length(g.boxmap.domain_points(g.boxmap.domain...).iter)
    print(io, "GPUSampledBoxMap with $(n) sample points")
end

function launch_kernel!(n, kernel, args...)
    compiled_kernel! = @cuda launch=false kernel(args...)
    config  = launch_configuration(compiled_kernel!.fun)
    threads = min(n, config.threads)
    blocks  = cld(n, threads)
    compiled_kernel!(args...; threads, blocks)
    return
end

function launch_kernel_then_sync!(n, kernel, args...)
    launch_kernel!(n, kernel, args...)
    CUDA.synchronize()
    return
end

function available_array_memory()
    m = CUDA.MemoryInfo()
    return m.free_bytes + m.pool_reserved_bytes
end

cu_reduce(::Type{I}) where {I<:Integer} = Int32
cu_reduce(::Type{Int16}) = Int16
cu_reduce(::Type{Int8}) = Int8
cu_reduce(::Type{F}) where {F<:AbstractFloat} = Float32
cu_reduce(::Type{Float16}) = Float16
cu_reduce(::Type{<:NTuple{N,T}}) where {N,T} = NTuple{N,cu_reduce(T)}

function out_of_bounds(::P) where {N,T,I,P<:BoxGrid{N,T,I}}
    K = cu_reduce(keytype(P))
    K(ntuple(_->0, Val(N)))
end

function out_of_bounds(::P) where {N,T,I,P<:BoxTree{N,T,I}}
    K = cu_reduce(keytype(P))
    K((0, ntuple(_->0, Val(N))))
end


function Adapt.adapt_structure(a::A, b::BoxGrid{N,T,I}) where {N,T,I,A<:Union{<:CUDA.CuArrayKernelAdaptor,<:CUDA.KernelAdaptor}}
    TT, II = cu_reduce(T), cu_reduce(I)
    Adapt.adapt_storage(a, 
        BoxGrid{N,TT,II}(
            Box{N,TT}(b.domain...),
            SVector{N,TT}(b.left),
            SVector{N,TT}(b.scale),
            SVector{N,II}(b.dims)
        )
    )
end

function Adapt.adapt_structure(a::A, b::BoxTree{N,T,I}) where {N,T,I,A<:Union{<:CUDA.CuArrayKernelAdaptor,<:CUDA.KernelAdaptor}}
    TT = cu_reduce(T)
    Adapt.adapt_storage(a,
        BoxTree(
            Box{N,TT}(b.domain...),
            Adapt.adapt(a, b.nodes)
        )
    )
end

function Adapt.adapt_structure(
        ::CUDA.CuArrayKernelAdaptor, x::V
    ) where {N,M,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F},M}}

    CuArray{SVector{N,cu_reduce(F)},M}(x)
end

# hotfix to avoid errors due to cuda device-side printing
@propagate_inbounds function point_to_key(partition::BoxGrid{N,T,I}, point, ::Val{:gpu}) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, partition, x_ints)
        @cuprint(
            "something went wrong in point_to_key. Fixing\n", 
            #="point:\n    $(point.data) \n", 
            "xi:\n    $(xi.data) \n", 
            "x:ints:\n    $(x_ints.data) \n", 
            "domain:\n    $(partition.domain.center.data)\n    $(partition.domain.radius.data) \n",
            "dims:\n    $(partition.dims.data) \n"=#
        )
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(partition))
    end
    return x_ints
end

end # module
