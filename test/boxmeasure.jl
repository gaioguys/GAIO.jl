using GAIO
using Test

@testset "exported functionality" begin
    domain = Box((0.0, 0.0), (1.0, 1.0))
    partition = BoxPartition(domain, (16,8))
    left  = cover(partition, Box((-0.5, 0.0), (0.5, 1.0)))
    right = cover(partition, Box((0.5, 0.0), (0.5, 1.0)))
    full  = cover(partition, :)

    n = length(right)
    @test length(left) == n
    @test length(full) == 2n

    scale = volume(domain) / 2n
    μ_left  = BoxMeasure(left, ones(n) .* scale)
    μ_right = BoxMeasure(right, ones(n) .* scale)
    μ_full  = BoxMeasure(full, ones(2n) .* scale)

    @testset "vector space structure" begin
        @test μ_left + μ_right == μ_full
        @test μ_full - μ_left == μ_right
        @test μ_left - μ_full == -μ_right
        @test 2*μ_left + 2*μ_right == μ_full + μ_full
        @test μ_left/2 + μ_right/2 == μ_full/2
    end

    h((x, y)) = (x+1, y)
    H = BoxMap(:sampled, h, domain, GAIO.center, GAIO.center)
    T = TransferOperator(H, full, full)

    @testset "applying transfer operator" begin
        @test T*μ_left == μ_right
        @test T'μ_right == μ_left
    end

    @testset "integration" begin
        @test μ_full(domain) == volume(domain)
        @test sum(x->2, μ_full) == 2*volume(domain)
        @test (2*μ_full)(domain) == 2*volume(domain)
    end
end
