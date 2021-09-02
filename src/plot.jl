include("plot/shader.jl")
include("plot/camera.jl")
include("plot/plot.jl")

function plot(boxset::BoxSet{<:AbstractBoxPartition{<:Box{N}}}; kwargs...) where N
    m = HyperRectangle(Vec3f0(0), Vec3f0(1))
    c = [box.center for box in boxset]
    r = [box.radius for box in boxset]
    fig, ax, ms = meshscatter(Vec{N, Float32}.(c), marker = m, markersize = 1.9*r,
                            color =:red; kwargs...)
end

function plot(boxfun::BoxFun{<:AbstractBoxPartition{<:Box{1}}}; kwargs...)
    c, r, v = Float32[], Float32[], Float32[]

    for (key, value) in boxfun.dict
        box = GAIO.key_to_box(boxfun.partition, key)
        push!(c, box.center[1])
        push!(r, box.radius[1])
        push!(v, value)
    end
    fig, ax, bp = barplot(c, v, width = 1.9*r; kwargs ...)
end

function plot(boxfun::BoxFun{<:AbstractBoxPartition{<:Box{3}}}; kwargs...) where N
    center, radius, color = Vec{3, Float32}[], Vec{3, Float32}[], Float32[]

    for (key, value) in boxfun.dict
        box = GAIO.key_to_box(boxfun.partition, key)
        push!(center, box.center)
        push!(radius, box.radius)
        push!(color, value)
    end
    m = HyperRectangle(Vec3f0(0), Vec3f0(1))
    fig = Figure()
    fig, ax, ms = meshscatter(center, marker = m, markersize = 1.9*radius, 
                              color = color, colormap =:jet; kwargs...)
    Colorbar(fig[1,2], ms)
    fig
end
