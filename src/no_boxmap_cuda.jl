# This file exists to provide docstrings for the Docs website

"""
    BoxMap(:gpu, map, domain; no_of_points) -> CPUSampledBoxMap

Transforms a ``map: Q → Q`` defined on points in 
the domain ``Q ⊂ ℝᴺ`` to a `CPUSampledBoxMap` defined 
on `Box`es. 

Uses the GPU's acceleration capabilities. 

By default uses a grid of sample points. 


    BoxMap(:sampled, :gpu, boxmap)
    GPUSampledBoxMap(boxmap)

Type representing a dicretization of a map using 
sample points, which are mapped on the gpu. This 
type performs orders of magnitude faster than 
standard `SampledBoxMap`s. 

!!! warning "`image_points` with `GPUSampledBoxMap`"
    `GPUSampledBoxMap` makes NO use of the `image_points` 
    field in `SampledBoxMap`s. 

Fields:
* `boxmap`:     `SampledBoxMap` with one restriction: 
                `boxmap.image_points` will not be used. 


Requires a CUDA-capable gpu. 
"""
struct GPUSampledBoxMap{N,T,F<:SampledBoxMap{N,T}} <: BoxMap
    boxmap::F
    GPUSampledBoxMap(boxmap) = @error "Did not find a CUDA-capable gpu."
end

"""
    BoxMap(:pointdiscretized, :gpu, map, domain, points) -> SampledBoxMap

Construct a `GPUSampledBoxMap` that uses the Vector `points` as test points. 
`points` must be a VECTOR of test points within the unit cube 
`[-1,1]^N`. 

Requires a CUDA-capable gpu. 
"""
function PointDiscretizedBoxMap(c::Val{:gpu}, map, domain::Box{N,T}, points) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end

"""
    BoxMap(:grid, :gpu, map, domain; no_of_points::NTuple{N} = ntuple(_->16, N)) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `no_of_points`, which is 
a tuple of length equal to the dimension of the domain. 

Requires a CUDA-capapble gpu.
"""
function GridBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points=ntuple(_->4*pick_vector_width(T),N)) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end
    
"""
    BoxMap(:montecarlo, :gpu, map, domain; no_of_points=16*N) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses `no_of_points` 
Monte-Carlo test points. 

Requires a CUDA-capapble gpu.
"""
function MonteCarloBoxMap(c::Val{:gpu}, map, domain::Box{N,T}; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end
