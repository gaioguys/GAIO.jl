using GAIO
using StaticArrays
using DynamicalSystems, OrdinaryDiffEq, BenchmarkTools

ENV["JULIA_DEBUG"] = "all"

# the Lorenz system
σ, ρ, β = 10.0, 28.0, 0.4
const p0 = SA_F64[σ,ρ,β]

τ, n_steps = 0.01, 20
const Δt = τ*n_steps

function lorenz_dudx(u::Vec, p, t, Δt) where Vec
    x,y,z = u
    σ,ρ,β = p

    dudx = Vec(( σ*(y-x), ρ*x-y-x*z, x*y-β*z ))
    return Δt * dudx
end

lorenz(u, p=p0, t=0) = lorenz_dudx(u, p, t, Δt)

# hand-coded integrator: most efficient (if possible)
f(x) = rk4_flow_map(lorenz, x, 1/20, 20)

# DynamicalSystem: less efficient but can use DifferentialEquations under the hood
u0 = rand(SVector{3,Float64})

alg = RK4(thread = OrdinaryDiffEq.False())
diffeq = (alg = alg, dt = 1/20, adaptive = false)
dyn_syst = ContinuousDynamicalSystem(lorenz, u0, p0; diffeq)

# DifferentialEquations integrator: least efficient but maximally customizable
prob = ODEProblem(lorenz, u0, (0.,1.), p0)

function h(x)
    new_prob = remake(prob, u0=x)
    solution = solve(
        new_prob, alg, dt = 1/20, 
        adaptive = false, save_everystep=false, save_start=false
    )
    return solution.u[end]
end

# analysis
center, radius = (0,0,25), (30,30,30)
domain = Box(center, radius)
P = BoxPartition(domain, (128,128,128))

x = ( sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1 )   # equilibrium
S = cover(P, x)

F = BoxMap(:grid, f, domain, n_points=(2,2,2))
G = BoxMap(:grid, dyn_syst, domain, n_points=(2,2,2))
H = BoxMap(:grid, h, domain, n_points=(2,2,2))

@benchmark F(S)
@benchmark G(S)
@benchmark H(S)

F = BoxMap(f, domain)
W = unstable_set(F, S)

#using Plots: plot
using GLMakie

plot(W)


# ---------------------------------------------------------


# the Henon map
a, b = 1.4, 0.3
u0 = SA_F64[0, 0]
const p0 = SA_F64[a, b]

function f(u::Vec, p=p0, t=0) where Vec
    x,y = u
    a,b = p
    return Vec(( 1 - a*x^2 + y, b*x ))
end

system = DiscreteDynamicalSystem(f, u0, p0)

center, radius = (0, 0), (3, 3)
domain = Box(center, radius)
P = BoxPartition(domain)

F = BoxMap(:grid, f, domain)
G = BoxMap(:grid, system, domain)

S = cover(P, :)

@benchmark relative_attractor(F, S, steps = 16)
@benchmark relative_attractor(G, S, steps = 16)

A = relative_attractor(F, S, steps = 16)

using Plots: plot
#using WGLMakie    # same result, just interactive

plot(Ā)
