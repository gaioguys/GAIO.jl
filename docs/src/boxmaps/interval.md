# `IntervalBoxMap`

All of the above techniques provide a fast, efficient way to cover setwise images of boxes, but are not necessarily guaranteed to provide an complete covering. To avoid this as well as other numerical inaccuracies inherent in floating point arithmetic, one can use _interval arithmetic_ to guarantee a rigorous outer covering of box images. Interval arithmetic is a technique from _validated numerics_ which performs calculations while simultaneously recording the error of such calculations. A more detailed discussion and julia-implementation of interval arithmetic can be found in [`IntervalArithmetic.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl). 

```@docs
IntervalBoxMap
```
