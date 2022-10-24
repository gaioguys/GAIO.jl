using LinearAlgebra, StaticArrays
using WGLMakie: plot
using GAIO

function Sdiagm(factor::T=1, size) where T
    SMatrix{size,size,T}(
        ntuple(
            i -> (i - 1) % (size + 1) ? factor : zero(T),
            size ^ 2
        )
    )
end

# (scaled) Dadras system
# we use a coordinate transformation x̃ = μ(x)
# with μ(x) = x * η(x), η(x) = 1 / (sqrt ∘ norm)(x)
const ee = 2 * eps() ^ (1/3)
const a, b, c = 8.0, 40.0, 14.9
function v(q::SVector{4,T}) where {T}
    #η = 1 / max((sqrt ∘ norm)(q), ee)   # to ensure we dont get Inf
    η = 1 / (sqrt ∘ norm)(q)
    ∇η = -q .* (η ^ 3) ./ 2
    #Dμ = Sdiagm(η) .+ ∇η
    vq = let x = q[1], y = q[2], z = q[3], w = q[4]
        SVector{4,T}(a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
    end
    #vq̃ = Dμ * vq
    vq̃ = η .* vq .+ q .* (∇η'vq)
end
f(x) = rk4_flow_map(v, x, 0.01, 1)

# The system is extremely expansive, so resolving the entire box image is 
# difficult. Hence we try with an adaptive test point sampling approach 
# that attempts to handle errors due to the map diverging. 

const montecarlo_points = [ SVector{4,Float64}(2f0*rand(Float64,4).-1f0 ...) for _ = 1:400 ]

function domain_points(center::SVector{N,T}, radius::SVector{N,T}) where {N,T}
    try 
        points = sample_adaptive(f, center, radius)
    catch ex
        ex isa Union{InterruptException,OutOfMemoryError} && rethrow(ex)
        @warn "$ex was thrown during adaptive sampling." box=Box{N,T}(center, radius)
        points = rescale(center, radius, montecarlo_points)
    end
end 

image_points(center, radius) = vertices(center, radius)

domain = Box((0,0,0,0), (250,150,200,25))

P = BoxPartition(domain, (128,128,128,128))
equillibrium = P[Box((0,0,0,0), (0.1,0.1,0.1,0.1))]

F = SampledBoxMap(
    f, 
    domain,
    domain_points,
    image_points,
    nothing
)
#F = BoxMap(f, P, no_of_points=40)

@profview W = unstable_set!(F, equillibrium)

P = BoxPartition(domain)
@profview W = relative_attractor(F, P[:], steps=20)


plot(W)

#T = TransferOperator(W)
#(λ, ev) = eigs(T)

#plot(log∘abs∘ev[1])
