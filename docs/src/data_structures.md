# Data Structures

```@index
Pages = ["data_structures.md"]
```

```@docs
Box
volume(box::Box)
```

```@docs
BoxPartition
subdivide(P::BoxPartition{N,T,I}, dim) where {N,T,I}
```

```@docs
TreePartition
subdivide!
```

```@docs
SampledBoxMap
BoxMap
PointDiscretizedMap
AdaptiveBoxMap
```

```@docs
BoxSet
```

```@docs
BoxFun
Base.sum(f, boxfun::BoxFun)
∘(f, boxfun::BoxFun)
```

```@docs
TransferOperator
```

Add `SparseArrays.sparse(g::TransferOperator)` once next PR is pushed
