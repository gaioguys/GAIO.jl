include("plot/shader.jl")
include("plot/camera.jl")
include("plot/plot.jl")

function plot(boxset::BoxSet{<:BoxPartition{<:Box{N}}}; kwargs...) where N
    if length(boxset) < 1e6
        m = HyperRectangle(Vec3f0(0), Vec3f0(1))
        c = [box.center for box in boxset]
        r = [box.radius for box in boxset]
        meshscatter(Vec{N, Float32}.(c), marker = m, markersize = 1.9*r; kwargs...)
    else
        plot([GLBox{N}(box.center.data, box.radius.data, (1.0f0, 1.0f0, 1.0f0, 1.0f0)) for box in boxset])
    end
end
