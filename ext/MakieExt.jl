module MakieExt

using GAIO, Makie, GeometryBasics, StaticArrays
using Makie: MakieCore
import GAIO: default_box_color
    
"""
    plot(boxset::BoxSet)
    plot(boxmeas::BoxMeasure)
    plot!(boxset::BoxSet)
    plot!(boxmeas::BoxMeasure)

Plot a `BoxSet` or `BoxMeasure`. 

## Special Attributes:

`projection = x -> x[1:3]`
If the dimension of the system is larger than 3, use this function to project to 3-d space.

`color = :red`
Color used for the boxes.

`colormap = :default`
Colormap used for plotting `BoxMeasure`s values.

`marker = HyperRectangle(GeometryBasics.Vec3f0(0), GeometryBasics.Vec3f0(1))`
The marker for an individual box. Only works if using Makie for plotting. 

All other attributes are taken from MeshScatter.

"""
@recipe(PlotBoxes) do scene
    attr = MakieCore.Attributes(
        marker     = HyperRectangle(GeometryBasics.Vec3f0(0), GeometryBasics.Vec3f0(1)),
        projection = nothing,
        color      = default_box_color
    )
    MakieCore.shading_attributes!(attr)
    MakieCore.generic_plot_attributes!(attr)
    MakieCore.colormap_attributes!(attr, MakieCore.theme(scene, :colormap))
end

Makie.preferred_axis_type(::PlotBoxes) = Axis3

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxSet{GAIO.Box{N,T}}}}) where {N,T}

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

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{N,T}}}}) where {N,T}

    boxmeas = boxes[1][]
    d = min(N, 3)
    if isnothing(boxes.projection[])
        boxes.projection[] = x -> x[1:d]
    end
    q = boxes.projection[]

    center = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxmeas))
    radius = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxmeas))
    colors = Vector{Float32}(undef, length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
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

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{2,T}}}}) where {T}

    boxmeas = boxes[1][]

    center = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxmeas))
    radius = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
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

function MakieCore.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{1,T}}}}) where {T}

    boxmeas = boxes[1][]

    height = Vector{Float32}(undef, 2*length(boxmeas))
    center = Vector{Float32}(undef, 2*length(boxmeas))
    radius = Vector{Float32}(undef, 2*length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
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

MakieCore.plottype(::Union{BoxSet,BoxMeasure}) = PlotBoxes

function MakieCore.convert_arguments(::MakieCore.PointBased, coords::AbstractVector{<:Complex})
    #Float32.(real.(coords)), Float32.(imag.(coords))
    (map(x -> Point2f0(real(x), imag(x)), coords),)
end

function MakieCore.convert_arguments(::MakieCore.PointBased, coords::AbstractVector{<:Complex}, heights::AbstractVector{<:Real})
    #Float32.(real.(coords)), Float32.(imag.(coords)), Float32.(heights)
    (map((x,y) -> Point3f0(real(x), imag(x), y)),)
end

MakieCore.plottype(::AbstractVector{<:Complex}) = Scatter

end # module
