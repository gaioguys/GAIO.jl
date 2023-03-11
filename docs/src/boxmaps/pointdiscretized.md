# `PointDiscretizedBoxMap`

A generalization of `MonteCarloBoxMap` and `GridBoxMap` can be defined as follows: 
1. we provide a "global" set of test points within the unit cube ``[-1,1]^d``. 
2. For each box `Box(c,r)`, we rescale the global test points to lie within the box by calculating `c .+ r .* p` for each global test point `p`. 

```@docs
PointDiscretizedBoxMap
```
