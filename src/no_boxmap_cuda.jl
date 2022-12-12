# This file exists to provide docstrings for the Docs website

"""
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
    PointDiscretizedBoxMap(map, domain, points, Val(:gpu)) -> SampledBoxMap

Construct a `GPUSampledBoxMap` that uses the Vector `points` as test points. 
`points` must be a VECTOR of test points within the unit cube 
`[-1,1]^N`. 

Requires a CUDA-capable gpu. 
"""
function PointDiscretizedBoxMap(map, domain::Box{N,T}, points, ::Val{:gpu}) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end

"""
GridBoxMap(map, domain, Val(:gpu); no_of_points::NTuple{N} = ntuple(_->16, N)) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses a grid of test points. 
The size of the grid is defined by `no_of_points`, which is 
a tuple of length equal to the dimension of the domain. 

Requires a CUDA-capapble gpu.
"""
function GridBoxMap(map, domain::Box{N,T}, c::Val{:gpu}; no_of_points=ntuple(_->4*pick_vector_width(T),N)) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end
    
"""
MonteCarloBoxMap(map, domain, Val(:gpu); no_of_points=16*N) -> GPUSampledBoxMap

Construct a `GPUSampledBoxMap` that uses `no_of_points` 
Monte-Carlo test points. 

Requires a CUDA-capapble gpu.
"""
function MonteCarloBoxMap(map, domain::Box{N,T}, c::Val{:gpu}; no_of_points=4*N*pick_vector_width(T)) where {N,T}
    @error "Did not find a CUDA-capable gpu."
end
