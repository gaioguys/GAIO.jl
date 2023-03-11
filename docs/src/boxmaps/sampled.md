# `SampledBoxMap`

We can even further generalize the concept of `MonteCarloBoxMap`, `GridBoxMap`, `PointDiscretizedBoxMap` as follows: we define two functions `domain_points(c, r)` and `image_points(c, r)` for any `Box(c, r)`. 
1. for each box `Box(c, r)` a set of test points within the box is initialized using `domain_points(C, r)` and mapped forward by the point map. 
2. For each of the pointwise images `fc`, an optional set of "perturbations" can be applied. These perturbations are generated with `image_points(fc, r)`. The boxes which are hit by these perturbations are recorded. 

```@docs
SampledBoxMap
CPUSampledBoxMap
GPUSampledBoxMap
```
