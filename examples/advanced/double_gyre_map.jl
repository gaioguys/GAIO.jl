using GAIO
using Plots
using LinearAlgebra
using ProgressMeter

#using Preferences
#set_preferences!(GAIO, "precompile_workload" => false; force=true)
default(size=(900,600), colormap=:jet)

#                defining the map
# -------------------------------------------------

const A, ϵ, ω = 0.25, 0.25, 2π

f(x, t)  =  ϵ * sin(ω*t) * x^2 + (1 - 2ϵ * sin(ω*t)) * x
df(x, t) = 2ϵ * sin(ω*t) * x   + (1 - 2ϵ * sin(ω*t))

double_gyre(x, y, t) = (
    -π * A * sin(π * f(x, t)) * cos(π * y),
     π * A * cos(π * f(x, t)) * sin(π * y) * df(x, t)
)

# autonomize the ODE by adding a dimension
double_gyre((x, y, t)) = (double_gyre(x, y, t)..., 1)

# nonautonomous flow map: reduce back to 2 dims
function φ((x, y), t, τ, steps)
    (x, y, t) = rk4_flow_map(double_gyre, (x, y, t), τ, steps)
    return (x, y)
end

t₀, τ, steps = 0, 0.1, 20
t₁ = t₀ + τ * steps
φₜ₀ᵗ¹(z) = φ(z, t₀, τ, steps)

#               GAIO.jl functions
# -------------------------------------------------

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxPartition(domain, (256, 128))
S = cover(P, :)
𝚽 = BoxMap(:grid, φₜ₀ᵗ¹, domain, n_points=(6,6))

Tspan = t₁ - t₀
γ = finite_time_lyapunov_exponents(𝚽, S; T=Tspan)

plot(γ, clims=(0,2))


𝚽♯ = TransferOperator(𝚽, S, S)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(𝚽♯, 2))
λ, ev = eigs(𝚽♯; which=:LR, maxiter=maxiter, tol=tol, v0=v0)

plot(real ∘ ev[2])


function rescale!(𝚽♯::TransferOperator)
    M = 𝚽♯.mat
    p = ones(size(𝚽♯, 2))
    q = M * p
    M .= Diagonal(1 ./ sqrt.(q)) * M
    𝚽♯
end

rescale!(𝚽♯)
U, σ, V = svds(𝚽♯; maxiter=maxiter, tol=tol, v0=v0)

plot(sign ∘ U[2])

# applying GAIO.jl functions to multiple start times
# to animate time-dependent results
# -------------------------------------------------

prog = Progress(length(t₀:τ:t₁))
anim1 = @animate for t in t₀:τ:t₁
    next!(prog)

    φₜ(z) = φ(z, t, τ, steps)
    
    𝚽 = BoxMap(:grid, φₜ, domain, n_points=(6,6))
    γ = finite_time_lyapunov_exponents(𝚽, S; T=Tspan)

    M = maximum(γ)
    γ = 1/M * γ

    plot(γ, clims=(0,1))
end
gif(anim1, fps=Tspan÷(2τ))


prog = Progress(length(t₀:τ:t₁))
anim2 = @animate for t in t₀:τ:t₁
    next!(prog)

    φₜ(z) = φ(z, t, τ, steps)

    𝚽 = BoxMap(:grid, φₜ, domain, n_points=(6,6))
    𝚽♯ = TransferOperator(𝚽, S, S)
    λ, ev = eigs(𝚽♯; which=:LR, maxiter=maxiter, tol=tol, v0=v0)

    μ = real ∘ ev[2]
    s = sign(μ[(65,65)])
    M = maximum(abs ∘ μ)
    μ = s/M * μ

    plot(μ, clims=(-1,1))
end
gif(anim2, fps=Tspan÷(2τ))


prog = Progress(length(t₀:τ:t₁))
anim3 = @animate for t in t₀:τ:t₁
    next!(prog)

    φₜ(z) = φ(z, t, τ, steps)

    𝚽 = BoxMap(:grid, φₜ, domain, n_points=(6,6))
    𝚽♯ = TransferOperator(𝚽, S, S)
    rescale!(𝚽♯)
    U, σ, V = svds(𝚽♯; maxiter=maxiter, tol=tol, v0=v0)

    μ = ( x -> sign(x) * log(abs(x) + 1e-4) ) ∘ U[2]
    s = sign(μ[(10,40)])
    M = maximum(abs ∘ μ)
    μ = s/M * μ

    plot(μ, clims=(-1,1))
end
gif(anim3, fps=Tspan÷(2τ))
