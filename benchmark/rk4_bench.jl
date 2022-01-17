using SIMD, MuladdMacro, LinearAlgebra, StaticArrays, BenchmarkTools
using LoopVectorization, Random, Plots, DelimitedFiles
Random.seed!(1234)

const sixth, third = 1/6, 1/3
const σ, ρ, β = 10.0, 28.0, 0.4

function v(u)
    let x=u[1], y=u[2], z=u[3]
        return (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
    end
end
function unrolled_v(u)
    @assert (nN = length(u)) % 3 == 0
    du = similar(u)
    @turbo for i in 0:3:nN-3
        du[i+1] =     σ * u[i+2]    -     σ * u[i+1]
        du[i+2] =     ρ * u[i+1]    -  u[i+1] * u[i+3] - u[i+2]
        du[i+3] =  u[i+1] * u[i+2]  -     β * u[i+3]
    end
    return du
end

benchmarks = Dict()
timeseries(bench) = map(b -> median(b).time, bench)

u_init(n) = copy(reinterpret(SVector{3,Float64}, 10. .* rand(3*n) .+ fill(10., 3*n)))
u_16 = u_init(16)

#= -------------------------------------------- =#
# serial

@muladd @inline function rk4(f, x, τ)
    τp2 = τ/2

    k  = f(x)
    dx = @. k * sixth
    k  = @. x + τp2 * k

    k  = f(k)
    dx = @. dx + k * third
    k  = @. x + τp2 * k

    k  = f(k)
    dx = @. dx + k * third
    k  = @. x + τ * k

    k  = f(k)
    dx = @. dx + k * sixth
    k  = @. x + τ * dx

    return k
end

@inbounds function rk4_flow_map(f, u; τ=0.01, steps=20)
    u_new = copy(u)
    for _ in 1:steps
        u_new = rk4(f, u_new, τ)
    end
    return u_new
end

rk4_control = rk4_flow_map.(v, copy(u_16))
benchmarks["serial"] = [@benchmark $(rk4_flow_map).($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("serial finished")
p = plot(
    8:8:128,
    timeseries(benchmarks["serial"]),
    lab="serial",
    xlabel="number of points",
    ylabel="time to eval (ns)",
    title="Benchmarks of various RK4 Implementations",
    size=(1200,900),
    leg=:outerbottom
);

#= -------------------------------------------- =#
# serial, inplace

@muladd @inline function rk4!(f, x, dx, k, τ, τp2)

       k .= f(x)
    @. dx = k * sixth
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τ * k

       k .= f(k)
    @. dx = dx + k * sixth
    @. x  = x + τ * dx

    return nothing
end

@inbounds function rk4_flow_map!(f, u; τ=0.01, steps=20)
    du, k = similar(u), similar(u)
    τp2 = τ/2
    for _ in 1:steps
        rk4!(f, u, du, k, τ, τp2)
    end
    return nothing
end

rk4_test = MVector{3,Float64}.(copy(u_16))
rk4_flow_map!.(v, rk4_test)
@assert rk4_test ≈ rk4_control
benchmarks["serial, inplace"] = [@benchmark $(rk4_flow_map!).($v, u) setup=(u=MVector{3,Float64}.($(u_init)($n))) for n in 8:8:128]
println("serial, inplace finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["serial"]),
    lab="serial, inplace"
);

#= -------------------------------------------- =#
# ILP, @turbo

@muladd @inline function rk4(f, x, τ, n, N)
    τp2 = τ/2
    k, dx = similar(x), similar(x)

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(x[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        k[i] = x[i] + τ * dx[i]
    end

    return k
end

@inbounds function rk4_flow_map(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20) where {N,T}
    n = length(u)
    u_new = reinterpret(T, u)
    for _ in 1:steps
        u_new = rk4(f, u_new, τ, n, N)
    end
    return reinterpret(SVector{N,T}, u_new)
end

rk4_test = rk4_flow_map(v, copy(u_16))
@assert rk4_test ≈ rk4_control
benchmarks["ILP with @turbo"] = [@benchmark $(rk4_flow_map)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("ILP with @turbo finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["ILP with @turbo"]),
    linestyle=:dash,
    lab="ILP with @turbo"
);

#= -------------------------------------------- =#
# ILP, @turbo, inplace

@muladd @inline function rk4!(f, x, dx, k, τ, τp2, n, N)

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(x[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    @inbounds for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        x[i] = x[i] + τ * dx[i]
    end

    return nothing
end

@inbounds function rk4_flow_map!(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20) where {N,T}
    n = length(u)
    τp2 = τ/2
    u_new = reinterpret(T, u)
    du, k = similar(u_new), similar(u_new)
    for _ in 1:steps
        rk4!(f, u_new, du, k, τ, τp2, n, N)
    end
    u = reinterpret(SVector{N,T}, u_new)
    return nothing
end

rk4_test = copy(u_16)
rk4_flow_map!(v, rk4_test)
@assert rk4_test ≈ rk4_control
benchmarks["ILP with @turbo, inplace"] = [@benchmark $(rk4_flow_map!)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("ILP with @turbo, inplace finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["ILP with @turbo, inplace"]),
    linestyle=:dashdot,
    lab="ILP with @turbo, inplace"
);

#= -------------------------------------------- =#
# ILP, @turbo, unrolled function

@muladd @inline function rk4(f, x, τ, n, N)
    τp2 = τ/2
    k, dx = similar(x), similar(x)

    k = f(x)
    @turbo for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    k = f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    k = f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    k = f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        k[i] = x[i] + τ * dx[i]
    end

    return k
end

@inbounds function rk4_flow_map(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20) where {N,T}
    n = length(u)
    u_new = reinterpret(T, u)
    for _ in 1:steps
        u_new = rk4(f, u_new, τ, n, N)
    end
    return reinterpret(SVector{N,T}, u_new)
end

rk4_test = rk4_flow_map(unrolled_v, copy(u_16))
@assert rk4_test ≈ rk4_control
benchmarks["ILP with @turbo, unrolled function"] = [@benchmark $(rk4_flow_map)($(unrolled_v), u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("ILP with @turbo, unrolled function finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["ILP with @turbo, unrolled function"]),
    linestyle=:dashdotdot,
    lab="ILP with @turbo, unrolled function"
);

#= -------------------------------------------- =#
# ILP, @turbo, unrolled function, inplace

@muladd @inline function rk4!(f, x, dx, k, τ, τp2, n, N)

    k .= f(x)
    @turbo for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    k .= f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    k .= f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    k .= f(k)
    @turbo for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        x[i] = x[i] + τ * dx[i]
    end

    return nothing
end

@inbounds function rk4_flow_map!(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20) where {N,T}
    n = length(u)
    τp2 = τ/2
    u_new = reinterpret(T, u)
    du = similar(u_new)
    k = copy(u_new)
    for _ in 1:steps
        rk4!(f, u_new, du, k, τ, τp2, n, N)
    end
    u = reinterpret(SVector{N,T}, u_new)
    return nothing
end

rk4_test = copy(u_16)
rk4_flow_map!(unrolled_v, rk4_test)
@assert rk4_test ≈ rk4_control
benchmarks["ILP with @turbo, unrolled function, inplace"] = [@benchmark $(rk4_flow_map!)($(unrolled_v), u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("ILP with @turbo, unrolled function, inplace finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["ILP with @turbo, unrolled function, inplace"]),
    linestyle=:dot,
    lab="ILP with @turbo, unrolled function, inplace"
);

#= -------------------------------------------- =#
# manually vectorized

@muladd @inline function rk4(f, x, τ)
    τp2 = τ/2

    k  = f(x)
    dx = @. k * sixth
    k  = @. x + τp2 * k

    k  = f(k)
    dx = @. dx + k * third
    k  = @. x + τp2 * k

    k  = f(k)
    dx = @. dx + k * third
    k  = @. x + τ * k

    k  = f(k)
    dx = @. dx + k * sixth
    k  = @. x + τ * dx

    return k
end

function rk4_flow_map(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20, simd_length=4) where {N, T}
    n = length(u)
    m, r = divrem(n, simd_length)
    @assert r == 0

    u_new = reshape(permutedims(reshape(reinterpret(T, u), (N, n))), (N*n,))
    idx = VecRange{simd_length}(1)
    u_tmp = Vector{Vec{simd_length,T}}(undef, N)
    @muladd @inbounds for i in 0:m-1
        u_tmp = [u_new[idx + simd_length*i + n*j] for j in 0:N-1]
        for _ in 1:steps
            u_tmp = rk4(f, u_tmp, τ)
        end
        for j in 0:N-1
            u_new[idx + simd_length*i + n*j] = u_tmp[j + 1]
        end
    end

    return copy(reinterpret(SVector{N,T}, reshape(permutedims(reshape(u_new, (n,N))), (N*n,))))
end

rk4_test = rk4_flow_map(v, copy(u_16))
@assert rk4_test ≈ rk4_control
benchmarks["manually vectorized"] = [@benchmark $(rk4_flow_map)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("manually vectorized finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["manually vectorized"]),
    marker=:circle,
    lab="manually vectorized"
);

#= -------------------------------------------- =#
# manually vectorized, inplace

@muladd @inline function rk4!(f, x, dx, k, τ, τp2)

       k .= f(x)
    @. dx = k * sixth
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τp2 * k

       k .= f(k)
    @. dx = dx + k * third
    @. k  = x + τ * k

       k .= f(k)
    @. dx = dx + k * sixth
    @. x  = x + τ * dx

    return nothing
end

function rk4_flow_map!(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20, simd_length=4) where {N, T}
    n = length(u)
    m, r = divrem(n, simd_length)
    @assert r == 0
    τp2 = τ/2

    u_new = reshape(permutedims(reshape(reinterpret(T, u), (N, n))), (N*n,))
    idx = VecRange{simd_length}(1)
    u_tmp = Vector{Vec{simd_length,T}}(undef, N)
    du, k = similar(u_tmp), similar(u_tmp)
    @muladd @inbounds for i in 0:m-1
        u_tmp = [u_new[idx + simd_length*i + n*j] for j in 0:N-1]
        for _ in 1:steps
            rk4!(f, u_tmp, du, k, τ, τp2)
        end
        for j in 0:N-1
            u_new[idx + simd_length*i + n*j] = u_tmp[j + 1]
        end
    end

    u .= copy(reinterpret(SVector{N,T}, reshape(permutedims(reshape(u_new, (n,N))), (N*n,))))
    return nothing
end

rk4_test = copy(u_16)
rk4_flow_map!(v, rk4_test)
@assert rk4_test ≈ rk4_control
benchmarks["manually vectorized, inplace"] = [@benchmark $(rk4_flow_map!)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("manually vectorized, inplace finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["manually vectorized, inplace"]),
    marker=:+,
    lab="manually vectorized, inplace"
);

#= -------------------------------------------- =#
# manually vectorized, ILP

@inbounds @muladd @inline function rk4(f, x, τ, n, N)
    τp2 = τ/2
    k, dx = similar(x), similar(x)

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(x[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        k[i] = x[i] + τ * dx[i]
    end

    return k
end

@inbounds function rk4_flow_map(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20, simd_length=4) where {N, T}
    n = length(u)
    m, r = divrem(n, simd_length)
    @assert r == 0

    u_new = reshape(permutedims(reshape(reinterpret(T, u), (N, n))), (N*n,))
    idx = VecRange{simd_length}(1)
    u_tmp = Vector{Vec{simd_length,T}}(
        reshape(
            [u_new[idx + simd_length*i + n*j]
             for j in 0:N-1, i in 0:m-1],
            (N*m,)
        )
    )

    for _ in 1:steps
        u_tmp = rk4(f, u_tmp, τ, m, N)
    end
    for i in 0:m-1, j in 0:N-1
        u_new[idx + simd_length*i + n*j] = u_tmp[N*i + j + 1]
    end

    return copy(reinterpret(SVector{N,T}, reshape(permutedims(reshape(u_new, (n,N))), (N*n,))))
end

rk4_test = rk4_flow_map(v, copy(u_16))
@assert rk4_test ≈ rk4_control
benchmarks["manually vectorized, ILP"] = [@benchmark $(rk4_flow_map)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("manually vectorized, ILP finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["manually vectorized, ILP"]),
    marker=:diamond,
    lab="manually vectorized, ILP"
);

#= -------------------------------------------- =#
# manually vectorized, ILP, inplace

@inbounds @muladd @inline function rk4!(f, x, dx, k, τ, τp2, n, N)

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(x[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = k[i] * sixth
        k[i] = x[i] + τp2 * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τp2 * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * third
        k[i] = x[i] + τ * k[i]
    end

    for i in 0:N:(n-1)*N
        k[i+1:i+N] .= f(k[i+1:i+N])
    end
    for i in 1:n*N
        dx[i] = dx[i] + k[i] * sixth
        x[i] = x[i] + τ * dx[i]
    end

    return nothing
end

@inbounds function rk4_flow_map!(f, u::Vector{SVector{N,T}}; τ=0.01, steps=20, simd_length=4) where {N, T}
    n = length(u)
    m, r = divrem(n, simd_length)
    @assert r == 0
    τp2 = τ/2

    u_new = reshape(permutedims(reshape(reinterpret(T, u), (N, n))), (N*n,))
    idx = VecRange{simd_length}(1)
    u_tmp = Vector{Vec{simd_length,T}}(
        reshape(
            [u_new[idx + simd_length*i + n*j]
             for j in 0:N-1, i in 0:m-1],
            (N*m,)
        )
    )
    du, k = similar(u_tmp), similar(u_tmp)

    for _ in 1:steps
        rk4!(f, u_tmp, du, k, τ, τp2, m, N)
    end
    for i in 0:m-1, j in 0:N-1
        u_new[idx + simd_length*i + n*j] = u_tmp[N*i + j + 1]
    end

    u .= copy(reinterpret(SVector{N,T}, reshape(permutedims(reshape(u_new, (n,N))), (N*n,))))
    return nothing
end

rk4_test = copy(u_16)
rk4_flow_map!(v, rk4_test)
@assert rk4_test ≈ rk4_control
benchmarks["manually vectorized, ILP, inplace"] = [@benchmark $(rk4_flow_map!)($v, u) setup=(u=$(u_init)($n)) for n in 8:8:128]
println("manually vectorized, ILP, inplace finished")
p = plot!(
    p,
    8:8:128,
    timeseries(benchmarks["manually vectorized, ILP, inplace"]),
    marker=:dtriangle,
    lab="manually vectorized, ILP, inplace"
);

#= -------------------------------------------- =#

key = collect(keys(benchmarks))
writedlm(
    "./RK4_benchmarks.csv",
    vcat(
        permutedims(key), 
        hcat(
            [timeseries(benchmarks[k]) for k in key]...
        )
    )
)

savefig(p, "./RK4_benchmarks.svg")
display(p)
