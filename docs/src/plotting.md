# Plotting 

GAIO.jl hooks into the Makie.jl plotting API. This means that one can use all the surrounding functionality of Makie, eg. `Axis`, `Axis3`, `Colorbar`, etc. 

## Makie Backends

To see a plot, one needs to load one of the Makie backends (see [makie documentation about backends](https://makie.juliaplots.org/stable/#first_steps)). We will use GLMakie, which uses OpenGL. Add GLMakie using the package manager:
```julia
pkg> add GLMakie
```
Load the GLMakie backend and some plotting tools with
```julia
using GLMakie: plot #, Axis3, Colorbar, etc...
```

!!! warning "A note on Namespaces"
    Makie and GAIO.jl both export the type `Box`. For this reason, it is recommended NOT to use 
    ```julia
    using GLMakie
    ```
    and instead only load the function names one needs from Makie. 

## Plotting `BoxSet`s and `BoxFun`s

To plot a `BoxSet` or `BoxFun` `b`, simply call 
```julia
plot(b)
```
The mutating function `plot!` is also available. All of Makie's plotting keyword arguments, such as `color`, `colormap`, etc. In addition, the keyword argument `projection_func` is used to project to 3-dimensional space if the dimension of the space ``d`` is greater than 3. By default, the function used is `x -> x[1:3]`. For an example using a custom projection function, eg. to plot the unstable set of the _dadras system_:
```julia
using GAIO
using GLMakie: plot

# the Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
f(x) = rk4_flow_map(v, x, 0.01, 5)

cen, rad = (0,0,0,0), (250,150,200,25)
P = BoxPartition(Box(cen, rad), (128,128,128,128))
F = AdaptiveBoxMap(f, P.domain)

x = zeros(4)        # equilibrium
W = unstable_set!(F, P[x])

A = [1 0 0 0;
     0 1 0 0;
     0 0 1 0]

plot(W, projection_func = x -> A*x)
```

```@docs
GAIO.plotboxes
GAIO.plotboxes!
```