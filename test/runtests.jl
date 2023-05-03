using Preferences
using Test
using SafeTestsets

using SIMD
using CUDA
using GAIO

@testset "GAIO.jl" begin

    @info "testing box.jl"
    @safetestset "Box" begin
        include("box.jl")
    end

    @info "testing partition_regular.jl"
    @safetestset "BoxPartition" begin
        include("partition_regular.jl")
    end

    @info "testing partition_tree.jl"
    @safetestset "TreePartition" begin
        include("partition_tree.jl")
    end

    @info "testing boxset.jl"
    @safetestset "BoxSet" begin
        include("boxset.jl")
    end

    @info "testing boxmap.jl"
    @safetestset "SampledBoxMap" begin
        include("boxmap.jl")
    end

    @info "testing boxmap_simd.jl"
    @safetestset "SampledBoxMap :simd" begin
        include("boxmap_simd.jl")
    end

    if CUDA.functional()
        @info "testing boxmap_cuda.jl"
        @safetestset "SampledBoxMap :gpu" begin
            include("boxmap_cuda.jl")
        end
    end

    @info "testing boxmap_interval.jl"
    @safetestset "IntervalBoxMap" begin
        include("boxmap_interval.jl")
    end

    @info "testing boxfun.jl"
    @safetestset "BoxFun" begin
        include("boxfun.jl")
    end

    @info "testing algorithms.jl"
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end

    @info "testing unstable_manifold.jl"
    @safetestset "Lorenz system" begin
        include("unstable_manifold.jl")
    end
    
    if CUDA.functional()
        @info "testing unstable_manifold_cuda.jl"
        @safetestset "Lorenz system" begin
            include("unstable_manifold_cuda.jl")
        end
    end
end
