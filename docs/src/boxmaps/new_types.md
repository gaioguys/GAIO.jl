# Creating you own BoxMap type

Subtypes of the abstract type `BoxMap` must have four restrictions:
1. There must be a `domain` field within the type, i.e.
   ```julia
   struct MyBoxMap{N,T}
       domain::Box{N,T}
       # other things ...
   end
   ```
2. There must be a method `map_boxes(g::MyBoxMap, source::BoxSet)` which computes the setwise image of `source` under `g` and returns a `BoxSet`. 
3. There must be a method `construct_transfers(g::BoxMap, domain::BoxSet, codomain::BoxSet)` which computes a dictionary-of-keys sparse matrix `mat` with `mat[(hit_key, source_key)] = weight` for the TransferOperator, where `hit_key ∈ codomain.set` and `source_key ∈ domain.set`. 
