using GAIO, StaticArrays, Test

@testset "exported functionality" begin
    # a toy example with an easy attractor to test how well all the components
    # fit together
    f(x::SVector{2,Float64}) = SVector(x[1],x[2]*0.5)
    test_points = [SVector(-1.,-1.),SVector(-1.,1.),
                   SVector(1.,-1.),SVector(1.,1.)]
    g = boxmap(f,test_points)

    n = 10
    domain = Box(SVector(0.,0.),SVector(1.,1.))
    partition = RegularPartition(domain)
    partition_at_depth_n = RegularPartition(domain,n)
    rga = relative_attractor(partition[:],g,n)
    unstable = unstable_set!(partition_at_depth_n[:],g)
    #ground truths attractor and unstable set
    x_axis = [SVector(0,x) for x in range(-1,1,length=100)]
    y_axis = [SVector(x,0) for x in range(-1,1,length=100)]
    gt_rga = partition_at_depth_n[y_axis]
    gt_unstable = partition_at_depth_n[x_axis]

    # make sure that the algorithms cover the ground truth
    @test Base.length(gt_rga) > 0
    @test Base.length(intersect(rga,gt_rga)) == Base.length(gt_rga)
    @test Base.length(gt_unstable) > 0
    @test Base.length(intersect(unstable,gt_unstable)) == Base.length(gt_unstable)
end
