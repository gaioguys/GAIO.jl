using GAIO
using LoopVectorization

# This example demonstrates how to get a ~3x speedup
# in your code using your CPU's SIMD capabilities.

N = 3
const σ, ρ, β = 10.0, 28.0, 0.4

# By default, GAIO is set up to accept functions of the form
function f(x)
    # Some map, here we use the Lorenz equation
    dx = [
           σ * x[2] -    σ * x[1],
           ρ * x[1] - x[1] * x[3] - x[2],
        x[1] * x[2] -    β * x[3]
    ]
    return dx
end

# Internally, GAIO calls this function on a set of test points
# within the domain. This means many function calls have to be made.
# To make these function calls able to run in parallel, we will use LoopVectorization.

# But first, we will define an 'in-situ' function, ie a function
# that alters a previously defined 'output' dx.

# ---- WARNING: GAIO DOES NOT ACCEPT FUNCTIONS OF THIS FORM ----
function f2(dx, x)
    dx[1] =    σ * x[2] -    σ * x[1]
    dx[2] =    ρ * x[1] - x[1] * x[3] - x[2]
    dx[3] = x[1] * x[2] -    β * x[3]
    return nothing
end

# To acheive the speedup, we will use the following template,
# with N being the dimenion of our domain:
function f_template(x)
    dx = similar(x)
    @turbo for i in 0 : N : length(x) - N
        # replace like this
        #  x[j] ->  x[i+j]
        # dx[j] -> dx[i+j]
    end
    return dx
end

# All we have done is replace all calls to 
# getindex(vector, j) with getindex(vector, i+j).
# In our Lorenz equation example, this looks like 
N = 3
function f(x)
    dx = similar(x)
    for i in 0 : N : length(x) - N
        dx[i+1] =      σ * x[i+2] -      σ * x[i+1]
        dx[i+2] =      ρ * x[i+1] - x[i+1] * x[i+3] - x[i+2]
        dx[i+3] = x[i+1] * x[i+2] -      β * x[i+3]
    end
    return dx
end

# GAIO comes builtin with the classic Runge-Kutta-4 integrator,
# which is dispached to accept this type of unrolled function as well.
F(x) = rk4_flow_map(f, x)

# From here, you can access use all of GAIO's functionality as regular,
# we just need to pass :cpu to the BoxMap command.
center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
G = BoxMap(F, P, :cpu)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
W = unstable_set!(G, P[x])

plot(W)