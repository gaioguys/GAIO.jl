# Data Structures

```@index
Pages = ["data_structures.md"]
```

```@docs; canonical=false
Box
volume(box::Box)
GAIO.center
GAIO.radius
```

```@docs; canonical=false
BoxGrid
point_to_key
bounded_point_to_key
key_to_box
point_to_box
subdivide(P::BoxGrid{N,T,I}, dim) where {N,T,I}
```

```@docs; canonical=false
BoxTree
subdivide!
depth
tree_search
find_at_depth
leaves
hidden_keys
```

```@docs; canonical=false
BoxSet
GAIO.neighborhood
```

```@docs; canonical=false
BoxMap
SampledBoxMap
AdaptiveBoxMap
PointDiscretizedBoxMap
GridBoxMap
MonteCarloBoxMap
IntervalBoxMap
```

```@docs; canonical=false
BoxMeasure
sum(f, μ::BoxMeasure{B,K,V,P,D}; init...) where {B,K,V,P,D}
∘(f, μ::BoxMeasure)

```

```@docs; canonical=false
TransferOperator
eigs(g::TransferOperator, B; kwargs...)
svds(g::TransferOperator; kwargs...)
key_to_index
index_to_key
```
