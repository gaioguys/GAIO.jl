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
point_to_key
bounded_point_to_key
key_to_box
point_to_box
subdivide(P::BoxPartition{N,T,I}, dim) where {N,T,I}
```

```@docs
TreePartition
subdivide!
```

```@docs
BoxSet
```

```@docs
BoxGraph
```
