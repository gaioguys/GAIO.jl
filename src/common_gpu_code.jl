"""
Much of the code for the two gpu extensions is identical. 
This macro generates the identical part of the code, 
with appropriate object names. 
"""
macro common_gpu_code(gpuname, converter, ArrayType, ArrayAdaptor, KernelAdaptor)
code = esc(quote



BoxMap(::Val{Symbol("GPUSampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(::Val{Symbol("gpusampled")}, args...; kwargs...) = GPUSampledBoxMap(args...; kwargs...)
BoxMap(accel::Val{:gpu}, args...; kwargs...) = BoxMap(Val(:grid), accel, args...; kwargs...)
    
struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

# Constructors are copied from CUDAExt
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


Requires a $($(gpuname))-capable gpu. 
"""
struct GPUSampledBoxMap{N,T,F<:Function} <: BoxMap
    map::F
    domain::Box{N,T}
    domain_points::$ArrayType{SVector{N,T}}

    function GPUSampledBoxMap{N,T,F}(map::F, domain::Box, domain_points::$ArrayType{SVector{N,T}}) where {N,T,F<:Function}
        new{N,T,F}(map, Box{N,T}(domain...), domain_points)
    end
end

function GPUSampledBoxMap(map::F, domain::Box, domain_points::$ArrayType{SVector{N,T}}) where {N,T,F<:Function}
    GPUSampledBoxMap{N,T,F}(map, domain, domain_points)
end

function GPUSampledBoxMap(boxmap::SampledBoxMap{N}) where {N}
    c, r = boxmap.domain
    points = collect(boxmap.domain_points(c, r))
    map!(x -> (x .- c) ./ r, points, points)    # send all points to [-1, 1]ᴺ
    GPUSampledBoxMap(boxmap.map, boxmap.domain, $converter(points))
end

function BoxMap(symb::Symbol, ::Val{:gpu}, args...; kwargs...)
    F = BoxMap(symb, args...; kwargs...)
    GPUSampledBoxMap(F)
end

# helper + compatibility functions
function Base.show(io::IO, g::GPUSampledBoxMap)
    n = length(g.domain_points)
    print(io, "GPUSampledBoxMap with $(n) sample points")
end

# hotfix to avoid errors due to cuda device-side printing
@propagate_inbounds function point_to_key(partition::BoxGrid{N,T,I}, point, ::Val{:gpu}) where {N,T,I}
    point in partition.domain || return nothing
    xi = (point .- partition.left) .* partition.scale
    x_ints = ntuple( i -> unsafe_trunc(I, xi[i]) + one(I), Val(N) )
    @boundscheck if !checkbounds(Bool, partition, x_ints)
        x_ints = min.(max.(x_ints, ntuple(_->one(I),Val(N))), size(partition))
    end
    return x_ints
end

function out_of_bounds(::P) where {N,T,I,P<:BoxGrid{N,T,I}}
    K = cu_reduce(keytype(P))
    K(ntuple(_->0, Val(N)))
end

function out_of_bounds(::P) where {N,T,I,P<:BoxTree{N,T,I}}
    K = cu_reduce(keytype(P))
    K((0, ntuple(_->0, Val(N))))
end

function Adapt.adapt_storage(::A, (c,r)::Box{N,T}) where {N,T,A<:Union{<:$ArrayAdaptor,<:$KernelAdaptor}}
    TT = cu_reduce(T)
    Box{N,TT}(c,r)
end

function Adapt.adapt_structure(a::A, b::BoxGrid{N,T,I}) where {N,T,I,A<:Union{<:$ArrayAdaptor,<:$KernelAdaptor}}
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

function Adapt.adapt_structure(a::A, b::BoxTree{N,T,I}) where {N,T,I,A<:Union{<:$ArrayAdaptor,<:$KernelAdaptor}}
    TT = cu_reduce(T)
    Adapt.adapt_storage(a,
        BoxTree(
            Box{N,TT}(b.domain...),
            Adapt.adapt(a, b.nodes)
        )
    )
end

function Adapt.adapt_structure(
        ::$ArrayAdaptor, x::V
    ) where {N,M,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F},M}}

    $ArrayType{SVector{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::$ArrayAdaptor, x::V
    ) where {N,M,F<:Integer,V<:AbstractArray{<:NTuple{N,F},M}}

    $ArrayType{NTuple{N,cu_reduce(F)},M}(x)
end

function Adapt.adapt_structure(
        ::$ArrayAdaptor, x::V
    ) where {N,M,H<:Integer,F<:Integer,V<:AbstractArray{<:Tuple{H,NTuple{N,F}},M}}

    $ArrayType{Tuple{H,NTuple{N,cu_reduce(F)}},M}(x)
end

cu_reduce(::Type{I}) where {I<:Integer} = Int32
cu_reduce(::Type{Int16}) = Int16
cu_reduce(::Type{Int8}) = Int8
cu_reduce(::Type{F}) where {F<:AbstractFloat} = Float32
cu_reduce(::Type{Float16}) = Float16
cu_reduce(::Type{<:NTuple{N,T}}) where {N,T} = NTuple{N,cu_reduce(T)}



end)


#display(code)
return code
end
