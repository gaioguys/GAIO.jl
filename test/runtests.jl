using Preferences, Base.Sys
using Test
using SafeTestsets

using SIMD
using ProgressMeter
using GAIO

if isapple() && ARCH === :aarch64
    using Metal
else
    using CUDA
end

ENV["JULIA_DEBUG"] = GAIO

@testset "GAIO.jl" begin
    
    @info "testing box.jl"
    @safetestset "Box" begin
        include("box.jl")
    end

    @info "testing partition_regular.jl"
    @safetestset "BoxGrid" begin
        include("partition_regular.jl")
    end

    @info "testing partition_tree.jl"
    @safetestset "BoxTree" begin
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
    
    if ( isapple() && ARCH === :aarch64 ) || CUDA.functional()
        @info "testing boxmap_gpu.jl"
        @safetestset "SampledBoxMap :gpu" begin
            include("boxmap_gpu.jl")
        end
    end
    
    @info "testing boxmap_interval.jl"
    @safetestset "IntervalBoxMap" begin
        include("boxmap_interval.jl")
    end

    @info "testing progressmeter.jl"
    @safetestset "Progrss Meter" begin
        include("progressmeter.jl")
    end

    @info "testing boxmeasure.jl"
    @safetestset "BoxMeasure" begin
        include("boxmeasure.jl")
    end

    @info "testing algorithms.jl"
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
    
    @info "testing transfer_operator.jl"
    @safetestset "TransferOperator" begin
        include("transfer_operator.jl")
    end

    if ( isapple() && ARCH === :aarch64 ) || CUDA.functional()
        @info "testing transfer_operator_gpu.jl"
        @safetestset "TransferOperator :gpu" begin
            include("transfer_operator.jl")
        end
    end
    
    @info "testing unstable_manifold.jl"
    @safetestset "Lorenz system" begin
        include("unstable_manifold.jl")
    end
    
    if ( isapple() && ARCH === :aarch64 ) || CUDA.functional()
        @info "testing unstable_manifold_gpu.jl"
        @safetestset "Lorenz system :gpu" begin
            include("unstable_manifold_gpu.jl")
        end
    end
end
