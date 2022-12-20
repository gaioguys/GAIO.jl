struct NumLiteral{T} end
Base.:(*)(x, ::Type{NumLiteral{T}}) where T = T(x)
const i32, ui32 = NumLiteral{Int32}, NumLiteral{UInt32}

"""
    BoxMap(:gpu, map, domain; no_of_points) -> CPUSampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `CPUSampledBoxMap` defined 
on `Box`es. 

Uses the GPU's acceleration capabilities. 

By default uses a grid of sample points. 


    BoxMap(:sampled, :gpu, boxmap)
    GPUSampledBoxMap(boxmap)

Type representing a dicretization of a map using 
sample points, which are mapped on the gpu. This 
type performs orders of magnitude faster than 
standard `SampledBoxMap`s. 

!!! warning "`image_points` with `GPUSampledBoxMap`"
    `GPUSampledBoxMap` makes NO use of the `image_points` 
    field in `SampledBoxMap`s. 

Fields:
* `boxmap`:     `SampledBoxMap` with one restriction: 
                `boxmap.image_points` will not be used. 

.
"""
struct GPUSampledBoxMap{N,T,F<:SampledBoxMap{N,T}} <: BoxMap
    boxmap::F

    function GPUSampledBoxMap(g::F) where {N,T,F<:SampledBoxMap{N,T}}
        if !( g.domain_points(g.domain...) isa Base.Generator{<:Union{<:CuArray,<:CuDeviceArray}} )
            throw(AssertionError("""
            GPU BoxMaps require one set of "global" test points in the 
            unit box `[-1,1]^N`. In particular, `g.domain_points(c, r)` 
            must return `rescale(c, r, points)` for a `CuArray` `points` 
            of test points. 
            """))
        end
        new{N,T,F}(g)
    end
end

function map_boxes_kernel!(g, P::BoxPartition{N,T,I}, domain_points, in_keys, out_keys) where {N,T,I}
    nk, np = Int32.((length(in_keys), length(domain_points)))
    ind = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x #- 1i32
    stride = gridDim().x * blockDim().x
    len = nk * np #- 1i32
    for i in ind : stride : len
        m, n = CartesianIndices((np, nk))[i].I
        p    = domain_points[m]
        key  = in_keys[n]
        box  = key_to_box(P, key)
        c, r = box
        fp   = g(@muladd p .* r .+ c)
        hit  = point_to_key(P, fp)
        out_keys[i] = isnothing(hit) ? ntuple(_->zero(I), Val(N)) : hit
    end
end

function map_boxes(
        G::GPUSampledBoxMap{N,T}, source::BoxSet{B,Q,S}
    ) where {N,T,B,Q<:BoxPartition,S}

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
    delete!(image, K(ntuple(_->0, Val(N))))
    return BoxSet(P, image)
end

function construct_transfers(
        G::GPUSampledBoxMap, source::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q<:BoxPartition,S<:OrderedSet}

    g = G.boxmap
    p = g.domain_points(g.domain...).iter
    np = length(p)
    keys = Stateful(source.set)
    P = source.partition
    K = cu_reduce(keytype(Q))
    D = Dict{Tuple{K,K},cu_reduce(T)}
    mat = D()
    variant_keys = S()
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
            mat = mat ⊔ ((hit,key) => 1)
        end
        union!(variant_keys, setdiff(out_cpu, source.set))
        CUDA.unsafe_free!(in_keys); CUDA.unsafe_free!(out_keys)
    end
    delete!(variant_keys, K(ntuple(_->0, Val(N))))
    return mat, variant_keys
end

# constructors
"""
    BoxMap(:pointdiscretized, :gpu, map, domain, points) -> SampledBoxMap
    PointDiscretizedBoxMap(Val(:gpu), map, domain, points) -> SampledBoxMap

Construct a `GPUSampledBoxMap` that uses the Vector `points` as test points. 
`points` must be a VECTOR of test points within the unit cube 
`[-1,1]^N`. 
"""
function PointDiscretizedBoxMap(::Val{:gpu}, map, domain::Box{N,T}, points) where {N,T}
    points_vec = cu(points)
    boxmap = PointDiscretizedBoxMap(map, domain, points_vec)
    GPUSampledBoxMap(boxmap)
end

"""
    BoxMap(:grid, :gpu, map, domain; no_of_points::NTuple{N} = ntuple(_->16, N)) -> GPUSampledBoxMap
    GridBoxMap(Val(:gpu), map, domain; no_of_points::NTuple{N} = ntuple(_->16, N)) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `no_of_points`, which is 
a tuple of length equal to the dimension of the domain. 
"""
function GridBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points=ntuple(_->no_default(T),N)) where {N,T}
    Δp = 2 ./ no_of_points
    points = SVector{N,T}[ Δp.*(i.I.-1).-1 for i in CartesianIndices(no_of_points) ]
    PointDiscretizedBoxMap(c, map, domain, points)
end

function GridBoxMap(c::Val{:gpu}, map, P::BoxPartition{N,T}; no_of_points=ntuple(_->no_default(T),N)) where {N,T}
    GridBoxMap(c, map, P.domain, no_of_points=no_of_points)
end

"""
    BoxMap(:montecarlo, :gpu, map, domain; no_of_points=16*N) -> GPUSampledBoxMap
    MonteCarloBoxMap(Val(:gpu), map, domain; no_of_points=16*N) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses `no_of_points` 
Monte-Carlo test points. 
"""
function MonteCarloBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points=no_default(N,T)) where {N,T}
    points = SVector{N,T}[ 2*rand(T,N).-1 for _ = 1:no_of_points ] 
    PointDiscretizedBoxMap(c, map, domain, points)
end 

function MonteCarloBoxMap(c::Val{:gpu}, map, P::BoxPartition{N,T}; no_of_points=no_default(N,T)) where {N,T}
    MonteCarloBoxMap(c, map, P.domain; no_of_points=no_of_points)
end

# helper + compatibility functions
function Base.show(io::IO, g::GPUSampledBoxMap)
    center, radius = g.domain
    n = length(g.domain_points(center, radius).iter)
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

function Adapt.adapt_structure(a::A, b::BoxPartition{N,T,I}) where {N,T,I,A<:Union{<:CUDA.CuArrayAdaptor,<:CUDA.Adaptor}}
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

function Adapt.adapt_structure(
        ::CUDA.CuArrayAdaptor, x::V
    ) where {N,F<:AbstractFloat,V<:AbstractArray{<:SVNT{N,F}}}

    CuArray{SVector{N,cu_reduce(F)},1}(x)
end
