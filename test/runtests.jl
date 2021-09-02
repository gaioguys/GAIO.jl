using GAIO
using Test
using SafeTestsets

@testset "GAIO.jl" begin
    @safetestset "Box" begin
        include("box.jl")
    end
    @safetestset "BoxMap" begin
        include("boxmap.jl")
    end
    @safetestset "BoxSet" begin
        include("boxset.jl")
    end
    @safetestset "BoxPartition" begin
        include("partition_regular.jl")
    end
    @safetestset "TreePartition" begin
        include("partition_tree.jl")
    end
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
end
