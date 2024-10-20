using GAIO, LinearAlgebra, ProgressMeter

const A, ϵ, ω = 0.25, 0.25, 2.0*π

f(x, t)  =  ϵ * sin(ω*t) * x^2 + (1 - 2ϵ * sin(ω*t)) * x
df(x, t) = 2ϵ * sin(ω*t) * x   + (1 - 2ϵ * sin(ω*t))

double_gyre(x, y, t) = (
    -π * A * sin(π * f(x, t)) * cos(π * y),
     π * A * cos(π * f(x, t)) * sin(π * y) * df(x, t)
)

# autonomize the ODE by adding a dimension
double_gyre((x, y, t)) = (double_gyre(x, y, t)..., 1)

# nonautonomous flow map: reduce back to 2 dims
function Φ((x, y), t, τ, steps)
    (x, y, t) = rk4_flow_map(double_gyre, (x, y, t), τ, steps)
    return (x, y)
end

t₀, τ, steps = 0, 0.1, 20
t₁ = t₀ + τ * steps
Tspan = t₁ - t₀
Φₜ₀ᵗ¹(z) = Φ(z, t₀, τ, steps)

domain = Box((1.0, 0.5), (1.0, 0.5))
P = GridPartition(domain, (256, 128))
S = cover(P, :)

n_frames = 120
times = range(t₀, t₁, length=n_frames)

tol, maxiter, v0 = eps()^(1/4), 1000, ones(256*128)
function rescale!(T::TransferOperator)
    M = T.mat
    p = ones(size(T, 2))
    q = M * p
    M .= Diagonal(1 ./ sqrt.(q)) * M
end

# -----------------------------------------------------------

using Plots

prog = Progress(length(times))
anim = @animate for t in times
    Φₜ(z) = Φ(z, t, τ, steps)

    F = BoxMap(:grid, Φₜ, domain, n_points=(6,6))
    γ = finite_time_lyapunov_exponents(F, S; T=Tspan)

    next!(prog)
    plot(γ, clims=(0,2), colormap=:jet)
end;
gif(anim, "../assets/ftle1.gif", fps=20)


prog = Progress(length(times))
anim = @animate for t in times
    Φₜ(z) = Φ(z, t, τ, steps)

    F = BoxMap(:grid, Φₜ, domain, n_points=(6,6))
    F♯ = TransferOperator(F, S, S)
    rescale!(F♯)

    global maxiter, tol, v0
    U, σ, V = svds(F♯; maxiter=maxiter, tol=tol, v0=v0)

    μ = U[2]

    # do some rescaling to get a nice plot
    μ = ( x -> sign(x) * log(abs(x) + 1e-4) ) ∘ μ
    s = sign(μ[(10,40)])
    M = maximum(abs ∘ μ)
    μ = s/M * μ

    next!(prog)
    plot(μ, clims=(-1,1), colormap=:redsblues)
end;
gif(anim, "../assets/coherent.gif", fps=20)


prog = Progress(length(times))
anim = @animate for t in times
    Φₜ(z) = Φ(z, t, τ, steps)

    F = BoxMap(:grid, Φₜ, domain, n_points=(6,6))
    F♯ = TransferOperator(F, S, S)
    λ, ev = eigs(F♯; which=:LR, maxiter=maxiter, tol=tol, v0=v0)

    μ = real ∘ ev[2]

    # do some rescaling to get a nice plot
    s = sign(μ[(65,65)])
    M = maximum(abs ∘ μ)
    μ = s/M * μ

    next!(prog)
    plot(μ, clims=(-1,1), colormap=:jet)
end;
gif(anim, "../assets/gyre_almost_inv.gif", fps=20)
