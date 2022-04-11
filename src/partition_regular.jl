"""
Internal data structure to partition a box

`domain`:       box defining the entire domain

`left`:         leftmost / bottom edge of the domain

`scale`:        1 / diameter of each box in the new partition (componentwise)

`dimsprod`:     for indexing the partition:
                key is counted up in first dimension first, then second dimension, etc... ie

                 1st dim →
                  * — * — * — *
            2nd   | 1 | 2 | 3 |
            dim   * — * — * — *
            ↓     | 4 | 5 | 6 |
                  * — * — * — *
.
"""
struct BoxPartition{N,T} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    left::SVector{N,T}
    scale::SVector{N,T}
    dims::SVector{N,Int}
    dimsprod::SVector{N,Int}
end

function BoxPartition(domain::Box{N,T}, dims::NTuple{N,Int}) where {N,T}
    dims = SVector{N,Int}(dims)
    left = domain.center .- domain.radius
    scale = dims ./ (2 .* domain.radius)
    # nr. of boxes / diameter of the domain == 1 / diameter of each box
    dimsprod_ = [SVector(1); cumprod(dims)]
    dimsprod = dimsprod_[SOneTo(N)]

    return BoxPartition(domain, left, scale, dims, dimsprod)
end

function BoxPartition(domain::Box{N,T}) where {N,T}
    dims = tuple(ones(Int64,N)...)
    BoxPartition(domain, dims)
end

BoxPartition(domain::Box{1,T}, dims::Int) where {T} = BoxPartition(domain, (dims,))

dimension(::BoxPartition{N,T}) where {N,T} = N

function subdivide(P::BoxPartition{N,T}, dim::Int) where {N,T}
    new_dims = ntuple(i -> P.dims[i]*(i==dim ? 2 : 1), N)
    return BoxPartition(P.domain, new_dims)
end

keytype(::Type{<:BoxPartition}) = Int
keys_all(partition::BoxPartition) = 1:prod(partition.dims)
# == 1 : partition.dimsprod[end] * partition.dims[end]

Base.size(partition::BoxPartition) = partition.dims.data # .data returns as tuple

function Base.show(io::IO, partition::BoxPartition) 
    print(io, "$(partition.dims[1])")
    for k = 2:length(partition.left)
        print(io, " x $(partition.dims[k])")
    end 
    print(io, " BoxPartition")
end

# TODO: replace with overloaded getindex
@muladd function key_to_box(partition::BoxPartition{N,T}, key::M) where M <: Union{Int, NTuple{N, Int}} where {N,T}
    dims = size(partition)
    radius = partition.domain.radius ./ dims
    left = partition.domain.center .- partition.domain.radius
    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)
    # start at leftmost box in the partition and move $key boxes right
    return Box(center, radius)
end 

@muladd function unsafe_point_to_ints(partition::BoxPartition, point)
    x = (point .- partition.left) .* partition.scale    
    # counts how many boxes x is away from left (componentwise)
    return map(xi -> Base.unsafe_trunc(Int, xi), x)
end

#function unsafe_point_to_ints(partition::BoxPartition, point::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
@muladd function unsafe_point_to_ints(
        partition::BoxPartition, point::SV
    ) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}

    x = (point .- partition.left) .* partition.scale
    x_ints = map(x) do xi
        convert(SIMD.Vec{simd, Int}, trunc(xi))
    end
    return x_ints
end

function ints_to_key(partition::BoxPartition, x_ints)
    if any(x_ints .< zero(eltype(x_ints))) || any(x_ints .>= partition.dims)
        @debug "point does not lie in the domain" x_ints partition.dims
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + 1
    return key
end

#function ints_to_key(partition::BoxPartition, x_ints::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
@inbounds function ints_to_key(
        partition::BoxPartition, x_ints::SV
    ) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    
    in_bounds = all.(
        tuple_vscatter(
            ( x_ints .>= zero(T) ) .& ( x_ints .< partition.dims )
        )
    )
    key = NTuple{simd,T}(sum(x_ints .* partition.dimsprod) + 1)
    return key[in_bounds]
end

function point_to_key(partition::BoxPartition, point)
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    return key
end
