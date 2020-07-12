using GAIO, SafeTestsets

@testset "GAIO.jl" begin
    @safetestset "Box" begin include("box.jl") end
    @safetestset "RegularPartition" begin include("partition_regular.jl") end
    @safetestset "BoxSet" begin include("boxset.jl") end
end
