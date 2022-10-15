using LinearAlgebra, StaticArrays, MuladdMacro
using WGLMakie: plot
using GAIO

# (scaled) Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
function f(x)
    fx = rk4_flow_map(v, x)
    fx = fx ./ sqrt(norm(fx) + 1)
end

# The system is extremely expansive, so resolving the entire box image is 
# difficult. Hence we try with an adaptive test point sampling approach 
# that attempts to handle errors due to the map diverging. 

const montecarlo_points = [ tuple(2f0*rand(Float32,4).-1f0 ...) for _ = 1:4000 ]

function domain_points(center::SVector{N,T}, radius::SVector{N,T}) where {N,T}
    L, y, n, h = zeros(T,N,N), MVector{N,T}(zeros(T,N)), MVector{N,Int}(undef), MVector{N,T}(undef)
    fc = f(center)
    for dim in 1:N
        y[dim] = radius[dim]
        fr = f(center .+ y)
        L[:, dim] .= abs.(fr .- fc) ./ radius[dim]
        y[dim] = zero(T)
    end
    if any(!isfinite, L)
        @warn(
            """The dynamical system diverges within the box. 
            Cannot calculate Lipschitz constant. 
            Returning Monte-Carlo test points.""",
            box=Box{N,T}(center, radius)
        )
        return (@muladd(center .+ radius .* point) for point in montecarlo_points)
    end
    try
        _, σ, Vt = svd(L)
        n .= ceil.(Int, σ)
        h .= 2.0 ./ (n .- 1)
        points = Iterators.map(CartesianIndices(ntuple(k -> n[k], Val(N)))) do i
            p = [n[k] == 1 ? zero(T) : (i[k] - 1) * h[k] - 1 for k in 1:N]
            p .= Vt'p
            @muladd p .= center .+ radius .* p
            sp = SVector{N,T}(p)
        end
        return points
    catch ex
        @warn "$ex was thrown when calculating adaptive grid. Using fallback grid." maxlog=100
        op = opnorm(L, Inf)
        h .= (2 * minimum(radius) / (isnan(op) ? 0.5f0 : op)) .* ones(T,N)
        n .= floor.(Int, radius ./ h)
        points = Iterators.map(CartesianIndices(ntuple(k -> -n[k]:n[k], Val(N)))) do i
            p = SVector{N,T}(Tuple(i))
            @muladd sp = center .+ radius .* p
        end
        return points
    end
end 

image_points(center, radius) = vertices(center, radius)

domain = Box((0,0,0,0), (250,150,200,25))
P = BoxPartition(domain, (128,128,128,128))

F = SampledBoxMap(
    f, 
    domain,
    domain_points,
    image_points,
    nothing
)

x = zeros(4)        # equilibrium
W = unstable_set!(F, P[x])

plot(W)

#T = TransferOperator(W)
#(λ, ev) = eigs(T)

#plot(log∘abs∘ev[1])
