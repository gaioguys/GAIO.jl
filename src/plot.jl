# include("plot/shader.jl")
# include("plot/camera.jl")
# include("plot/plot.jl")

const default_box_color = :red

"""
    plot(boxset::BoxSet)
    plot(boxfun::BoxFun)
    plot!(boxset::BoxSet)
    plot!(boxfun::BoxFun)

Plot a `BoxSet` or `BoxFun`. 

## Special Attributes:

`projection = x -> x[1:3]`
If the dimension of the system is larger than 3, use this function to project to 3-d space.

`color = :red`
Color used for the boxes.

`colormap = :default`
Colormap used for plotting `BoxFun`s values.

`marker = HyperRectangle(GeometryBasics.Vec3f0(0), GeometryBasics.Vec3f0(1))`
The marker for an individual box. Only works if using Makie for plotting. 

All other attributes are taken from MeshScatter.

"""
@recipe(PlotBoxes) do scene
    MakieCore.merge!(
        MakieCore.Attributes(
            marker     = HyperRectangle(GeometryBasics.Vec3f0(0), GeometryBasics.Vec3f0(1)),
            projection = nothing,
            color      = default_box_color
        ),
        MakieCore.default_theme(scene, MakieCore.MeshScatter)
    )
end

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxSet{Box{N,T}}}}) where {N,T}

    boxset = boxes[1][]
    d = min(N, 3)
    if isnothing(boxes.projection[])
        boxes.projection[] = x -> x[1:d]
    end
    q = boxes.projection[]

    center = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxset))
    radius = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxset))

    for (i, box) in enumerate(boxset)
        center[i] = q(box.center)
        radius[i] = q(box.radius) .* 1.9
    end

    MakieCore.meshscatter!(
        boxes, 
        center, 
        marker      = boxes.marker[], 
        color       = boxes.color[], 
        markersize  = radius
    )
end

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxFun{Box{N,T}}}}) where {N,T}

    boxfun = boxes[1][]
    d = min(N, 3)
    if isnothing(boxes.projection[])
        boxes.projection[] = x -> x[1:d]
    end
    q = boxes.projection[]

    center = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxfun))
    radius = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxfun))
    colors = Vector{Float32}(undef, length(boxfun))

    for (i, (box, value)) in enumerate(boxfun)
        center[i] = q(box.center)
        radius[i] = q(box.radius) .* 1.9
        colors[i] = value
    end

    boxes.color[] == default_box_color && (boxes.color[] = colors)
    boxes.colorrange[] = extrema(colors)

    MakieCore.meshscatter!(
        boxes, 
        center, 
        marker      = boxes.marker[], 
        colormap    = boxes.colormap[],
        color       = boxes.color[], 
        markersize  = radius
    )
end

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxFun{Box{2,T}}}}) where {T}

    boxfun = boxes[1][]

    center = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxfun))
    radius = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxfun))

    for (i, (box, value)) in enumerate(boxfun)
        center[2*i-1] = SVector{3,Float32}(box.center..., 0.)
        center[2*i]   = SVector{3,Float32}(box.center..., value)
        radius[2*i-1] = SVector{3,Float32}(box.radius..., minimum(box.radius))
        radius[2*i]   = radius[2*i-1]
    end

    boxes.colorrange[] = extrema(x -> x[3], center)

    MakieCore.meshscatter!(
        boxes, 
        center, 
        marker      = boxes.marker[], 
        color       = boxes.color[], 
        markersize  = radius
    )
end

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxFun{Box{1,T}}}}) where {T}

    boxfun = boxes[1][]

    height = Vector{Float32}(undef, 2*length(boxfun))
    center = Vector{Float32}(undef, 2*length(boxfun))
    radius = Vector{Float32}(undef, 2*length(boxfun))

    for (i, (box, value)) in enumerate(boxfun)
        height[2*i-1] = 0.
        height[2*i]   = value
        center[2*i-1] = box.center[1]
        center[2*i]   = center[2*i-1]
        radius[2*i-1] = box.radius[1] * 1.9
        radius[2*i]   = radius[2*i-1]
    end

    boxes.colorrange[] = extrema(height)

    MakieCore.linesegments!(
        boxes, 
        center,
        height,
        color      = boxes.color[],
        linewidth  = radius .* 1f3
    )
end

MakieCore.plottype(::Union{BoxSet,BoxFun}) = PlotBoxes

function MakieCore.convert_arguments(::MakieCore.PointBased, coords::AbstractVector{<:Complex})
    #Float32.(real.(coords)), Float32.(imag.(coords))
    (map(x -> Point2f0(real(x), imag(x)), coords),)
end

function MakieCore.convert_arguments(::MakieCore.PointBased, coords::AbstractVector{<:Complex}, heights::AbstractVector{<:Real})
    #Float32.(real.(coords)), Float32.(imag.(coords)), Float32.(heights)
    (map((x,y) -> Point3f0(real(x), imag(x), y)),)
end

MakieCore.plottype(::AbstractVector{<:Complex}) = Scatter

RecipesBase.@recipe function plot!(boxset::BoxSet{Box{N,T}}; projection=x->x[1:2]) where {N,T}
    xs = Vector{Float32}(undef, 5*length(boxset))
    ys = Vector{Float32}(undef, 5*length(boxset))
    q = projection
    
    for (i, box) in enumerate(boxset)
        c, r = q(box.center), q(box.radius)
        lo, hi = c .- r, c .+ r
        xs[5*(i-1)+1:5*i] .= (lo[1], hi[1], hi[1], lo[1], NaN)
        ys[5*(i-1)+1:5*i] .= (lo[2], lo[2], hi[2], hi[2], NaN)
    end
    
    seriestype := :shape
    color --> default_box_color
    linecolor --> default_box_color
    linewidth --> 0.
    
    xs, ys
end

RecipesBase.@recipe function plot!(boxset::BoxFun{Box{N,T}}; projection=x->x[1:2]) where {N,T}
    xs = Vector{Float32}(undef, 5*length(boxset))
    ys = Vector{Float32}(undef, 5*length(boxset))
    cs = Vector{Float32}(undef, 5*length(boxset))
    q = projection
    
    for (i, (box, val)) in enumerate(boxset)
        c, r = q(box.center), q(box.radius)
        lo, hi = c .- r, c .+ r
        xs[5*(i-1)+1:5*i] .= (lo[1], hi[1], hi[1], lo[1], NaN)
        ys[5*(i-1)+1:5*i] .= (lo[2], lo[2], hi[2], hi[2], NaN)
        cs[5*(i-1)+1:5*i] .= (val, val, val, val, NaN)
    end
    
    seriestype := :shape
    fill_z := cs
    line_z := cs

    xs, ys
end

RecipesBase.@recipe function plot!(boxset::BoxSet{Box{1,T}}) where {T}
    xs = Vector{Float32}(undef, 3*length(boxset))
    ys = zeros(3*length(boxset))
    
    for (i, box) in enumerate(boxset)
        c, r = box.center, box.radius
        lo, hi = c .- r, c .+ r
        xs[3*(i-1)+1:3*i] .= (lo[1], hi[1], NaN)
    end
    
    seriestype := :shape
    color --> default_box_color
    linecolor --> default_box_color
    
    xs, ys
end

RecipesBase.@recipe function plot!(boxset::BoxFun{Box{1,T}}) where {T}
    xs = Vector{Float32}(undef, 5*length(boxset))
    ys = Vector{Float32}(undef, 5*length(boxset))
    
    for (i, (box, val)) in enumerate(boxset)
        c, r = box.center, box.radius
        lo, hi = c .- r, c .+ r
        xs[5*(i-1)+1:5*i] .= (lo[1], hi[1], hi[1], lo[1], NaN)
        ys[5*(i-1)+1:5*i] .= (0, 0, val, val, NaN)
    end
    
    seriestype := :shape
    color --> default_box_color
    linecolor --> default_box_color

    xs, ys
end

#=
RecipesBase.@recipe function plot!(coords::AbstractVector{<:Complex})
    seriestype --> scatter
    float.(real.(coords)), float.(imag.(coords))
end

RecipesBase.@recipe function plot!(coords::AbstractVector{<:Complex}, heights::AbstractVector{<:Real})
    seriestype --> scatter
    float.(real.(coords)), float.(imag.(coords)), float.(heights)
end
=#
