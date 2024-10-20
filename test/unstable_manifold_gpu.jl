using GAIO
using StaticArrays
using Test

using Base.Sys
if isapple() && ARCH === :aarch64
    using Metal
else
    using CUDA
end

@testset "exported functionality" begin
    # the Lorenz system
    v((x,y,z)) = SA_F32[10f0*(y-x), 28f0*x-y-x*z, x*y-0.4f0*z]
    f(x) = rk4_flow_map(v, x, 0.01f0, 20)

    cen = SA_F32[0,0,25]
    rad = SA_F32[30,30,30]
    domain = Box{3,Float32}(cen, rad)
    σ=10f0; ρ=28f0; β=0.4f0
    x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
    P = BoxPartition(domain, (128,128,128))
    S = cover(P, x)

    @testset "gpu montecarlo" begin
        F = BoxMap(:montecarlo, :gpu, f, domain)
        W = unstable_set(F, S)
        @test W isa BoxSet # passes if no error is thrown
        #@info "length of unstable set gpu montecarlo" W

        @info "benchmark run gpu montecarlo"
        @time W = unstable_set(F, S);

        T = TransferOperator(F, W)
        T = TransferOperator(F, W, W)
        λ, ev, nconv = eigs(T, nev=1, tol=100*eps())
        @test ev[1] isa BoxMeasure # passes if no error is thrown
    end
end
