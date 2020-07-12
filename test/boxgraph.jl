using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    p = RegularPartition(Box([0.0], [1.0]), 1)
    g = boxmap(x -> SVector(0.5), [0.0])
    set = p[:]
    @test set.set == Set([1, 2])
    list = BoxList(set.partition, [1, 2])
    G = transition_graph(g, list)

    scc = strongly_connected_components(G)
    @test length(scc.set) == 1
    box_scc = first(scc.set)
    @test GAIO.key_to_box(p, box_scc).center == SVector(0.5)

    @test matrix(G) == [
        0.0 0.5
        0.0 0.5
    ]
end
