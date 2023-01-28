using GAIO
using Plots

#                defining the map
# -------------------------------------------------

const A, ϵ, ω = 0.25, 0.25, 2π
function double_gyre(x, y, t)
    f(x, t) = ϵ * sin.(ω * t) .* x .^ 2 .+ (1 .- 2ϵ * (sin.(ω * t))) .* x
    df(x, t) = 2 * ϵ * sin.(ω * t) .* x .+ (1 .- 2ϵ * (sin.(ω * t)))

    return (
        -π * A * sin.(π * f(x, t)) .* cos.(π * y),
        π * A * cos.(π * f(x, t)) .* sin.(π * y) .* df(x, t),
        1
    )
end

double_gyre((x, y, z)) = double_gyre(x, y, z)

# create a point map ℝ² → ℝ² that 
# integrates the vector field 
# with fixed start time t₀, step size τ 
# until a fixed end time t₁ is reached
function f((x₀, y₀), t₀, τ, t₁)
    z = (x₀, y₀, t₀)
    for _ in t₀ : τ : t₁-τ
        z = rk4(double_gyre, z, τ)
    end
    (x₁, y₁, t₁) = z
    return (x₁, y₁)
end

f(t₀, τ, t₁) = z -> f(z, t₀, τ, t₁)

#               GAIO.jl functions
# -------------------------------------------------

domain = Box((1.0, 0.5), (1.0, 0.5))
P = BoxPartition(domain, (256, 128))
S = cover(P, :)

t₀, τ, t₁ = 0, 0.1, 1
Tspan = t₁ - t₀
F = BoxMap(
    :montecarlo, 
    f(t₀, τ, t₁), 
    domain
)

γ = finite_time_lyapunov_exponents(F, S; T=Tspan)
plot(γ, clims=(0,2))

T = TransferOperator(F, S, S)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(T, 2))
λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

plot(ev[2])

# applying GAIO.jl functions to multiple time spans
# to animate time-dependent results
# -------------------------------------------------

anim = @animate for t in t₀:τ:t₁
    F = BoxMap(
        :montecarlo, 
        f(t, τ, t+Tspan), 
        domain
    )

    γ = finite_time_lyapunov_exponents(F, S; T=Tspan)
    plot(γ, clims=(0,2))
end
gif(anim, fps=Tspan÷τ)

anim2 = @animate for t in t₀:τ:t₁
    F = BoxMap(
        :montecarlo, 
        f(t, τ, t+Tspan), 
        domain
    )

    T = TransferOperator(F, S, S)
    λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)
    plot(abs ∘ ev[2], clims=(-0.05, 0.05))
end
gif(anim2, fps=Tspan÷τ)
