using GAIO, SafeTestsets

@safetestset "GAIO.jl" begin
    @safetestset "Box" begin include("box.jl") end
    @safetestset "RegularPartition" begin include("partition_regular.jl") end
    @safetestset "treePartition" begin include("partition_tree.jl") end
    @safetestset "BoxSet" begin include("boxset.jl") end
    @safetestset "BoxMap" begin include("boxmap.jl") end
end
