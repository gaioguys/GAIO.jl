using GAIO
using Test

@testset "exported functionality" begin
    # the Lorenz system
    σ, ρ, β = 10.0, 28.0, 0.4
    v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
    f(x) = rk4_flow_map(v, x)

    center, radius = (0,0,25), (30,30,30)
    domain = Box(center, radius)
    x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
    P = BoxPartition(domain, (128,128,128))
    S = cover(P, x)

    for (name, args) in Dict(
        "gpu montecarlo"        => (:montecarlo, :gpu),
        "gpu grid"              => (:grid, :gpu)
    )

        @testset "$name" begin
            F = BoxMap(args..., f, domain)
            W = unstable_set(F, S)
            @test W isa BoxSet  # passes if no error is thrown
            
    
            @info "benchmark run $name"
            @time W = unstable_set(F, S);
        
            T = TransferOperator(F, W)
            λ, ev, nconv = eigs(T, nev=1)
            @test ev[1] isa BoxFun  # passes if no error is thrown    
        end

    end
end
