# An Overview of BoxMap Types

There are multiple techniques one could use to discretize point maps into maps over boxes. In [General Usage](https://gaioguys.github.io/GAIO.jl/general/) the discretization `BoxMap` was already briefly introduced. 

```@docs
BoxMap
```

We now introduce a set of `BoxMap` subtypes for different discretization algorithms. The types fit into a heirarchy described in the diagram below. 

![Type Hierarchy](../assets/type_tree.jpg)

We will work from the "bottom up", starting with specific types that are of practical use, and then generalizing these approaches for the reader who wishes to know more. 
