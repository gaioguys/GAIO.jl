# Plotting 

GAIO.jl offers plotting recipes for both Plots.jl and Makie.jl. This means that one can use all the surrounding functionality of Plots.jl or makie.jl, e.g. creating multiple subplots, animations, colorbars, etc. simply by loading either Plots.jl or Makie.jl. 

!!! note "Why offer both Plots and Makie recipes?"
    Makie.jl offers much more funcitonality and performance in interactive plots, but suffers much more greatly from the well known [time to fist plot problem](https://discourse.julialang.org/t/roadmap-for-a-faster-time-to-first-plot/22956). Hence GAIO.jl offers Plots.jl recipes for fast, 2-dimensional, non-interactive plotting, and Makie.jl recipes for interactive, 2- and 3-dimensional, or publication-quality visualizations. To see a difference in the plotting capability, see [the Hénon map example](https://github.com/gaioguys/GAIO.jl/blob/master/examples/invariant_measure_2d.jl). 

## Using Plots

To make a plot, one needs to simply load the package. Add Plots using the package manager
```julia
pkg> add Plots
``` 
Load Plots with
```julia
using Plots
```

## Using Makie

To see a plot, one needs to 
* load MakieCore (this is necessary, otherwise the plot recipes won't load)
* load one of the Makie backends (see [makie documentation about backends](https://makie.juliaplots.org/stable/#first_steps)). 
We will use GLMakie, which uses OpenGL. Add the packages using the package manager
```julia
pkg> add MakieCore, GLMakie
```
Load the GLMakie backend and some plotting tools with
```julia
using MakieCore
using GLMakie: plot #, Axis3, Colorbar, etc...
```

!!! warning "A note on Namespaces"
    Makie and GAIO.jl both export the type `Box`. For this reason, it is recommended NOT to use 
    ```julia
    using GLMakie
    ```
    and instead only load the function names one needs from Makie. 

## Common Interface

To plot a `BoxSet` or `BoxFun` `b`, simply call 
```julia
plot(b)
```
The mutating function `plot!` is also available. All of Plots.jl's or Makie.jl's keyword arguments, such as `color`, `colormap`, etc. can be used. In addition, the keyword argument `projection` is used to project to a lower dimensional space if the dimension of the space ``d`` is greater than 2 for Plots.jl or greater than 3 for Makie.jl. By default, the function used is `x -> x[1:2]` for Plots.jl and `x -> x[1:3]` for Makie.jl. For an example using a custom projection function, eg. to plot the unstable set of the _dadras system_:
```julia
using GAIO
using MakieCore
using GLMakie: plot

# the Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
f(x) = rk4_flow_map(v, x, 0.01, 5)

cen, rad = (0,0,0,0), (250,150,200,25)
P = BoxPartition(Box(cen, rad), (128,128,128,128))
F = BoxMap(f, P.domain)

x = zeros(4)        # equilibrium
W = unstable_set(F, P[x])

A = [0 1 0 0;
     0 0 1 0;
     0 0 0 1]

plot(W, projection = x -> A*x)
```
