using GAIO
using CUDA
using Test
using SafeTestsets

@testset "GAIO.jl" begin
    @safetestset "Box" begin
        include("box.jl")
    end
    @safetestset "BoxPartition" begin
        include("partition_regular.jl")
    end
    @safetestset "TreePartition" begin
        include("partition_tree.jl")
    end
    @safetestset "BoxSet" begin
        include("boxset.jl")
    end
    @safetestset "SampledBoxMap" begin
        include("boxmap.jl")
    end
    @safetestset "SampledBoxMap :simd" begin
        include("boxmap_simd.jl")
    end
    if CUDA.functional()
        @safetestset "SampledBoxMap :gpu" begin
            include("boxmap_cuda.jl")
        end
    end
    @safetestset "IntervalBoxMap" begin
        include("boxmap_interval.jl")
    end
    @safetestset "BoxFun" begin
        include("boxfun.jl")
    end
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
    @safetestset "Lorenz system" begin
        include("unstable_manifold.jl")
    end
end
