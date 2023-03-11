# `AdaptiveBoxMap`

The above approaches may not necessarily be effective for covering the setwise image of a box. For choosing test points more effectively, we can use some knowledge of the Lipschitz matrix for ``f`` in a box `Box(c, r)`, that is, a matrix ``L \in \mathbb{R}^{d \times d}`` such that 
```math
| f(y) - f(z) | \leq L \, | y - z | \quad \text{for all } y, z \in \text{Box}(c, r),
```
where the operations ``| \cdot |`` and `` \leq `` are to be understood elementwise. The function `AdaptiveBoxMap` attempts to approximate ``L`` before choosing an adaptive grid of test points in each box, as described in [1]. 

```@docs
AdaptiveBoxMap
```

### References

[1] Oliver Junge. “Rigorous discretization of subdivision techniques”. In: _International Conference on Differential Equations_. Ed. by B. Fiedler, K. Gröger, and J. Sprekels. 1999.
