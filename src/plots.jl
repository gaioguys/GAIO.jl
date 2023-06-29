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
    linecolor --> :black
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
    fillcolor --> :match
    fill_z := cs
    #line_z := cs
    linecolor --> :black
    linewidth --> 0.

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
    linecolor --> :black
    linewidth --> 0.
    
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
    linecolor --> :black
    linewidth --> 0.

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
