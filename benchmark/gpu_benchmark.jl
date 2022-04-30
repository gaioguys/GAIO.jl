using GAIO
using StaticArrays
using CUDA
using GPUArrays
using Base.Cartesian: @nexprs, @ntuple, @nextract
using Base.Threads

N, T = 3, Float32
n = 2^27
x = CUDA.rand(T, N*n)
dx = CUDA.zeros(N*n)

const σ, ρ, β = 10.0f0, 28.0f0, 0.4f0
function f(x)
    dx = (
           σ * x[2] -    σ * x[1],
           ρ * x[1] - x[1] * x[3] - x[2],
        x[1] * x[2] -    β * x[3]
    )
    return dx
end
F(x) = rk4_flow_map(f, x)

box = Box((0f0, 0f0, 0f0), (1f0, 1f0, 1f0))

@generated function initiate_kernel(dx, x, F, c::SVector{N,T}, r::SVector{N,T}) where {N,T}
    return quote
        ind    = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1  # index of thread, 0-based.
        stride = gridDim().x * blockDim().x                             # jump the length of the grid.
        for i  = $N * ind : $N * stride : length(x) - $N                # each point is N units long.
            xi = @ntuple( $N, j -> x[i + j] )                           # gather the point - faster than Base.ntuple.
            xi = F(xi)                                                  # map xi using F.
            @nexprs( $N, j -> dx[i + j] = xi[j] )                       # scatter xi into dx,
        end                                                             # @nexprs auto unrolls broadcasted
    end                                                                 # setindex! call
end

CUDA.@device_code_warntype( @cuda launch=false initiate_kernel(dx, x, F, box.center, box.radius) ) 

kernel = @cuda launch=false initiate_kernel(dx, x, F, box.center, box.radius)
config = launch_configuration(kernel.fun)
threads = min(n, config.threads)
blocks = cld(n, threads)

x = CUDA.rand(T, N*n)
dx = CUDA.zeros(N*n)
kernel(dx, x, F, box.center, box.radius; threads, blocks)
@time begin
    kernel(dx, x, F, box.center, box.radius; threads, blocks)
    CUDA.synchronize()
end
Array(dx)
Array(x)

x_r = reinterpret(SVector{N,T}, Array(x))
dx_r = reinterpret(SVector{N,T}, Array(dx))
@time @threads for i in 1:length(x_r)
    dx_r[i] = F(x_r[i])
end



function key_to_cr(
        dims::SVector{N,T}, center::SVector{N,T}, radius::SVector{N,T}, key::M
    ) where {N,T,M<:Union{Int32, NTuple{N, Int32}}}

    l = center .- radius
    r = radius ./ dims
    c = l .+ r .+ (2 .* r) .* (CartesianIndices(dims.data)[key].I .- 1)
    return (c, r)
end

@muladd @generated function map_boxes!(G, keys_in, points, p::V, keys_out) where {N,T,V<:AbstractGPUArray{NTuple{N,T}}}
    return quote
        ind    = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
        stride = gridDim().x * blockDim().x
        @nextract( $N, p, i -> view(p, $N * (i-1) + 1 : $N * i) )
        nk, np = length(keys_in), length(points)
        for i in ind : stride : nk * np - 1
            n, m = divrem(i, np)
            ci = CartesianIndices(p_4)[keys_in[n + 1]].I
            c  = @. p_1 + p_2 + 2 * p_2 * (ci - 1)
            fp = points[m + 1] .* p_2 .+ c
            fp = G(fp)
            fp = (fp .- l) .* p_3
            ip = unsafe_trunc.(Int32, fp)
            if any(ip .< zero(Int32)) || any(ip .>= dims)
                continue
            end
            keys_out[sum(ip .* p_5) + 1] = true
        end
        return
    end
end 

function map_boxes(g, source)
    if isnothing(source.cache)
        P = source.partition
        c1 = cu([
            P.domain.center .- P.domain.radius;
            P.domain.radius ./ P.dims;
            P.scale;
            P.dims;
            P.dimsprod
        ])
        c2 = CuArray{Bool,1}(undef, P.dims[end] * P.dimsprod[end])
        c3 =   Array{Bool,1}(undef, P.dims[end] * P.dimsprod[end])
        source.cache = (c1, c2, c3) 
    end 
    k = CuArray(collect(source.set))
    p = g.domain_points(P.domain.center, P.domain.radius)
    c = source.cache
    CUDA.fill!(source.cache[2], false)
    @cuda threads= blocks= map_boxes!(g.map, k, p, c[1], c[2])
    copyto!(c[3], c[2])
    return BoxSet(P, Set{Int}(findall(c[3])))
end