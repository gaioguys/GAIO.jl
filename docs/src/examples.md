# Examples
In the following, we will showcase some of the algorithms GAIO is capable of based on one example each.


## Relative Global Attractor of the Hénon System 
The Hénon map is the two-dimensional quadratic map 
```math 
\begin{aligned} 
&x_{n+1} = 1-\alpha x_n^2+y_n \\
&y_{n+1} = \beta x_n 
\end{aligned} 
```

characterizing the Hénon system. The so called *classical Hénon map* has the parameters ``\alpha = 1.4`` and ``\beta = 0.3 ``. 
In the following, we will demonstrate how to compute the *relative global attractor* for the classical Hénon map.
Let us start by generating ``n`` points on each face of the square ``[-1,1]^2 \subset \mathbb{R}^2`` as well as initializing the Hénon map ``f`` and the boxmap `g` corresponding to the dynamics of ``f``
```julia
function henon()
    generate_points = n -> [
        [(x, -1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0) for x in LinRange(-1, 1, n)];
        [(-1.0, x) for x in LinRange(-1, 1, n)];
        [( 1.0, x) for x in LinRange(-1, 1, n)];
    ]

    f = x -> SVector(1 - 1.4*x[1]^2 + x[2], 0.3*x[1])
       
    g = PointDiscretizedMap(f, generate_points(20))
```
    
In order to create the boxset which is one of the input parameters of the function `relative_attractor`, we first need to initialize the domain, that is the box corrsponding to ``[-1,1]^2``, and how/if it is already partitioned. It is natural to choose a regular partition which is initialized with the whole domain on depth ``0``, that is the partition is not yet subdivided.
```julia
    partition = RegularPartition(Box(SVector(0.0, 0.0), SVector(2.0, 1.0)))
    boxset = boxset_full(partition)
```
Finally we are able to call `relative_attractor` by
```julia
    return relative_attractor(boxset, g, 20)
```
which will return the boxset corresponding to the relative global attractor we receive after subdividing the domain twenty times. We can review the result by plotting the attractor with the `plot`-command.

```jldoctest
function henon()
    n = 20

    points = [
        [(x, -1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0) for x in LinRange(-1, 1, n)];
        [(-1.0, x) for x in LinRange(-1, 1, n)];
        [( 1.0, x) for x in LinRange(-1, 1, n)];
    ]

    f = x -> SVector(1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])

    g = PointDiscretizedMap(f, points)
    partition = RegularPartition(Box(SVector(0.0, 0.0), SVector(1.0, 1.0)))
    boxset = partition[:]

    return relative_attractor(boxset, g, 20)
end

# output

plot(henon())
```
> [insert beautiful plot here]

## Chain Recurrent Set for the Knotted Flow Map
## Root Covering
## Unstable Manifold for the Lorenz System
Let us consider the *Lorenz System*, which is the following three-dimensional continuous system 
```math
\begin{aligned} 
&\frac{\mathcal{d}x}{\mathcal{d}t} = s(y-x) \\ 
&\frac{\mathcal{d}y}{\mathcal{d}t} = rx - y-xz \\
&\frac{\mathcal{d}z}{\mathcal{d}t} = xy - bz. \\ 
\end{aligned}
```
In this example, we will choose the parameter values as ``s = 10, r = 28, b = 0.4`` and we are looking for the *unstable manifold* through the equilibrium point ``x_0 = (\sqrt{b(r-1)}, \sqrt{b(r-1)}, r-1 ) ``, which is a subset of the Lorenz attractor.

In order to compute this, we first need to initialize the function ```lorenz_f```, which we will achieve by solving the Lorenz System with a Runge-Kutta 4th order ODE solver. 
By choosing an equally spaced covering of the cube `` [-1,1]^3`` we can initialize the boxmap `g`:
```julia
    grid = LinRange(-1, 1, 7)
    points = collect(Iterators.product(grid, grid, grid))
    g = PointDiscretizedMap(lorenz_f, points)
```
Now we can choose the domain, which has to contain the fixed point and for this algorithm needs to be subdivided already. We decide to subdivide the domain 24 times, which will give us a regular partition with ``2^{24}`` boxes.
```julia
    domain = Box(SVector(0.0, 0.0, 27.0), SVector(30.0, 30.0, 40.0))
    partition = RegularPartition(domain, 24)
```
We then define the starting point, which is the equilibrium point we mentioned earlier and initialize the target set, which at the moment contains only the small box containing the fixed point but will store all the boxes we collect in the course of the algorithm and will eventually contain the unstable manifold.
```julia
    rh = 28.0
    b = 0.4
    x0 = (sqrt(b*(rh-1)), sqrt(b*(rh-1)), rh-1)

    boxset = partition[x0]
```
The only thing left to do now is call the function `unstable_set`
```julia
    unstable_set!(boxset, g)
```

> [insert breathtaking plot here]
## Transition Matrix for the Lorenz System
