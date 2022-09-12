# Algorithms and Mathematical Background
!!! note "Note"
    In the following, let ``Q \subset \mathbb{R}^d`` be compact. Further, ``f : \mathbb{R}^d \to \mathbb{R}^d`` will always refer to the map describing the dynamics of a system, while `F` will be the corresponding `BoxMap`.
    
## Mapping Boxes

### Mathematical Background
The following algorithms all require a method to approximate the set-wise image ``f(b)`` of a box ``b``. To do this, GAIO.jl splits the domain ``Q`` into a partition `P` of boxes, and uses test points within ``b``. When a SampledBoxMap is initialized, we require a function to calculate test points that are mapped by ``f``. This function is stored in the field `F.domain_points`. These test points are then mapped forward by ``f``, and the boxes which are hit become the image set. More precisely, mapping a box set is done in two main steps within GAIO.jl: 
1. Test points within the box are generated (or retrieved) using `F.domain_points(b.center, b.radius)`. These test points are mapped forward by the given function `f`.
2. For each mapped test point `fp`, an optional set of ”perturbations” are generated using `F.image_points(fp, b.radius)`. For each of the perturbed points, the index of the box within the partition containing this point is calculated. This index gets added to the image set.

### Implementation
```julia
function map_boxes(F::BoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
    P = source.partition
    @floop for box in source
        c, r = box.center, box.radius
        for p in F.domain_points(c, r)
            fp = F.map(p)
            hitbox = point_to_box(P, fp)
            isnothing(hitbox) && continue
            r = hitbox.radius
            for ip in F.image_points(fp, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                @reduce(image = union!(S(), hit))
            end
        end
    end
    return BoxSet(P, image)
end 
```

## Relative Global Attractor

### Mathematical Background
The set 
```math
A_Q = \bigcap_{k \geq 0} f^k(Q)
```
is called the global attractor relative to ``Q``.
The relative global attractor can be seen as the set which is eventually approached by every orbit originating in ``Q``. In particular, ``A_Q`` contains each invariant set in ``Q`` and therefore all the potentially interesting dynamics. 
The idea of the algorithm is to cover the relative global attractor with boxes and recursively tighten the covering by refining appropriately selected boxes.

Mathematically, the algorithm to compute the global attractor relative to ``Q`` takes two input arguments: a compact set ``Q`` as well as a map ``f``, which describes the dynamics. Now in each iteration, two steps happen:
1. **subdivision step:** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved in `B`. 
2. **selection step:** All those boxes `b` in the new box set `B` whose image does not intersect the domain, ie ``f(b) \cap \left( \bigcup_{b' \in B} b' \right) \neq \emptyset``, get discarded. 

If we repeatedly refine the box set `B` through ``k`` subdivision steps, then as ``k \to \infty`` the collection of boxes ``B`` converges to the relative global attractor ``A_Q`` in the Hausdorff metric.

### Implementation
```julia
function relative_attractor(F::BoxMap, B::BoxSet{Box{N,T}}; steps=12) where {N,T}
    for k = 1:steps
        B = subdivide(B, (k % N) + 1)
        B = B ∩ F(B)
    end
    return B
end
```
The third input parameter `steps` describes the level of approximation. Since in each step of the algorithm the initial domain is divided in half, the final partition after `steps` many steps will contain ``n := 2^{\text{depth}}`` boxes, i.e. every box in the final covering is ``\frac{1}{n}`` times the size of the initial box. 
For this algorithm the box set should be the full partition of the set ``Q``. 

## Unstable Set

In the following we are presenting the algorithm to cover invariant manifolds within some domain ``Q``, which has to contain a fixed point.
!!! note "Note"
    For simplicity, we will explain the algorithm for the case of the *unstable manifold*. However one can compute the stable manifold as well by considering the boxmap describing the inverse map ``f^{-1}`` as input argument for the algorithm.

### Mathematical Background
The unstable manifold is defined as
```math
W^U(x_0) = \{x: \lim_{k \to - \infty} f^k(x) = x_0 \}
```
where ``x_0`` is a fixed point of ``f``.

The idea behind the algorithm to compute the unstable manifold can be explained in two steps. Before starting we need to identify a hyperbolic fixed point and the region ``Q``, which we are going to compute the manifold in. The region ``Q`` needs to be already partitioned into small boxes.
1. **initialization step** Since a fixed point is always part of the unstable manifold, we need to identify a small region/box containing this fixed point. This box may be known a-priori, or one can use the `relative_attractor` around a region where one suspects a fixed point to exist. 
2. **continuation step** The small box containing the fixed point is then mapped forward by `F` and the boxes that are hit under the image are added to the box collection. Then those newly included boxes are mapped forward and the procedure is repeated until no new boxes are added. 

!!! warning "Note on Convergence"
    One might not be able to compute the parts of the unstable manifold whose preimage lies outside the domain ``Q``.
    Thus, it is important to choose ``Q`` large enough.

### Implementation
```julia
function unstable_set!(F::BoxMap, B::BoxSet)
    B_new = B
    while !isempty(B_new)
        B_new = F(B_new)
        setdiff!(B_new, B)
        union!(B, B_new)
    end
    return B
end
```
The input argument `B` includes two things:

The domain ``Q`` we are going to compute the unstable manifold in (``Q`` can be implemented as a large `Box`) and the underlying partition of the domain. Unlike in the previous algorithm, the domain will not be subdivided along the algorithms course, but we need to pass a partition which is already subdivided to the depth ``d`` (and therefore the level of accuracy) we want our final boxcovering to have. 

Note: This algorithm works with two mutable sets of boxes: `B`, which collects the boxes we aquire in each iteration and will eventually cover part of the unstable manifold, and `B_new`, which will be overwritten in each iteration and contains only the boxes which will be newly added to our collection.

## Chain Recurrent Set

### Mathematical Background
The _chain recurrent set over ``Q``_ ``R_Q`` is defined as the set of all ``x_0 \in Q`` such that for every ``\epsilon > 0`` there exists a set 
```math 
\left\{ x_0,\, x_1,\, x_2,\, \ldots,\, x_{n-1} \right\} \subset Q \quad \text{with} \quad \| f(x_{i \, \text{mod} \, n}) - x_{i+1 \, \text{mod} \, n} \| < \epsilon \,\ \text{for all} \,\ i
```
The chain recurrent set describes "arbitrarily small perturbations" of periodic orbits. This definition is useful since our box coverings our finite and hence inherently slightly uncertain. 

The idea for the algorithm is to construct a directed graph ``G`` whose vertices are the box set ``B``, and for which edges are drawn from ``B_1`` to ``B_2`` if ``f(B_1) \cap B_2 \neq \emptyset``. We can now ask for a subset of the vertices, for which each vertex is part of a directed cycle. This set is equivalent to the _strongly connected subset of ``G``_. We therefore perform two steps: 
1. **subdivision step** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved in `B`. 
2. **graph construction step** Generate the graph `G`. This is done by generating the _transition matrix over `B`_ (see the next algorithm) and noting the nonzero elements. This is the adjacency matrix for the graph `G`. 
3. **selection step** Find the strongly connected subset of `G`. Discard all vertices (boxes) which are not part of a strongly connected component. 

If we repeadetly refine the strongly connected box set through ``k`` subdivision steps, then the algorithm converges to the chain recurrent set as ``k \to \infty`` in the Hausdorff metric. 

### Implementation
```julia
function chain_recurrent_set(F::BoxMap, B::BoxSet{Box{N,T}}; steps=12) where {N,T}
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        P = TransferOperator(F, B)
        G = Graph(P)
        B = strongly_connected_components(G)
    end
    return B
end
```

## Transfer Operator

### Mathematical Background
The transition matrix is the discretization of the _transfer operator ``P`` w.r.t. ``f``_. Formally, the transfer operator w.r.t. ``f`` is defined for measurable functions ``g`` implicitly through the integral equation
```math
\int_A Pg (x) \, d\mu (x) = \int_{f^{-1}(A)} g(x) \, d\mu (x) \quad \text{for any} \ \ A \ \ \text{measurable}
```
We will use a Galerkin approximation for ``P`` which maintains the eigenvalues and cyclic behavior of ``P``. To do this, we project to a subspace ``\chi_B`` generated by the basis ``\left\{ \chi_b\ \vert\ b \in B \right\}`` 
of indicator functions on the boxes of our box set. Further, we enumerate the partition `P = {b_1, b_2, ..., b_n}` with integer indices and define the _transition matrix_ 
```math
    (P^n)_{ij} = \frac{\mathcal{L}\left(b_j \cap f^{-1}(b_i)\right)}{\mathcal{L}(b_j)}, \quad i,\, j = 1, \ldots, n,
```
where ``\mathcal{L}`` is the lebesque measure. Finally, we define the approximate transfer operator ``Q_n P : \chi_B \to \chi_B`` as the linear extension of 
```math
    (Q_n P)\, \chi_{b_i} = \sum_{j = 1}^n P_{ij}^n\, \chi_{b_j}, \quad i = 1, \ldots, n.
```
The operator ``Q_n P`` can be created in GAIO.jl by calling 
```julia
T = TransferOperator(F, B)
```
where `F` is a `BoxMap` and `B` is a box set. `T` acts as a matrix in every way, but the explicit transition matrix ``P^n_{ij}`` can be generated by calling 
```julia
M = sparse(T)
```
To realize this approximation, we need to calculate ``P^n_{ij}``. For this there are two techniques discussed in [1]. The simpler of the two techniques is a Monte-Carlo approach. Namely, we choose a fixed number ``r`` of test points in one of the boxes ``b_j``, and set ``P^n_{ij}`` as the fraction of test points which land in ``b_i``. 

It is important to note that `TranferOperator` is only supported over the box set `B`, but if one lets a `TranferOperator` act on a `BoxFun` (see general), then the support `B` is extended "on the fly" to include the support of the `BoxFun`.

### Implementation
```julia
const plusmerge! = mergewith!(+)
function TransferOperator(
        g::BoxMap, boxset::BoxSet{B,Q,S}
    ) where {N,T,I,B,Q<:BoxPartition{N,T,I},S}

    P = boxset.partition
    D = Dict{Tuple{I,I},T}      # initialize a "dict-of-keys" sparse matrix
    @floop for key in boxset.set
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        domain_points = g.domain_points(c, r)
        inv_n = 1. / length(domain_points)
        for p in domain_points
            c = g.map(p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            r = hitbox.radius
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                @reduce(mat = plusmerge!(D(), D((key,hit) => inv_n)))
            end
        end
    end
    matcsc = sparse(mat, length(P), length(P))      # convert to "compressed sparse column" format
    return TransferOperator(g, boxset, matcsc)
end
```

## Root Covering

### Mathematical Background
Nonlinear optimization theory offers a multitude of algorithms to iteratively approximate roots of functions ``h : \mathbb{R}^d \to \mathbb{R}^d``, that is, algorithms ``f : \mathbb{R}^d \to \mathbb{R}^d`` such that (under some conditions) ``f^k (x) \to x_0`` as ``k \to \infty`` with ``h(x_0) = \mathbf{0}``. We can consider these algorithms from the point of view of dynamics, and reframe the problem of finding a root of ``h`` to finding a fixed point of ``f``. 

Specifically, we will consider ``f`` to be a globalized Newton algorithm. One step of the (local) Newton algorithm follows the specification: solve the linear equation 
```math
J_h (x) d = - h(x)
```
and set 
```math
f(x) = x + d, 
```
where ``J_h (x)`` is the Jacobi matrix of ``h`` at ``x``. 

The local Newton algorithm is not guaranteed to converge to a global solution to ``h(x) = 0``. To rectify this, the step size ``\| d \|`` and direction ``d / \| d \|`` need to be modified. There are multiple heuristics to do this, and GAIO.jl uses the "Armijo rule": fix some ``\sigma < 1`` and find the largest ``\alpha \leq 1`` such that 
```math
h(x + \alpha d) - h(x) \leq \alpha \sigma \, J_h (x)^T d.
```
This is done by initializing ``\alpha = 1`` and testing the above condition. If it is not satisfied, scale ``\alpha`` by some constant ``\rho``, ie set ``\alpha = \rho \cdot \alpha``, and test the condition again. GAIO.jl uses ``\sigma = 10^{-4}`` and ``\rho = 4 / 5``. 

Using this iterative solver, one can follow a technique very similar to the algorithm for the realtive attractor. 
1. **subdivision step:** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved in `B`. 
2. **selection step:** The box set `B` is mapped forward using one step of the adaptive newton algorithm. 

If we repeadetly refine the box set `B` through ``k`` subdivision steps, then as ``k \to \infty`` the collection of boxes converges to the set of roots of `h` in the Hausdorff metric. 

### Implementation
```julia
function cover_roots(h, Dh, B::BoxSet{Box{N,T}}; steps=12) where {N,T}
    domain = B.partition.domain
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)
        f = x -> adaptive_newton_step(h, Dh, x, k)
        F = BoxMap(f, domain)
        B = F(B)
    end
    return B
end
```
The arguments needed are the map `h`, some approximation of the Jacobian ``J_h`` which we call `Dh`, and a box set `B` containing some root of `h`. 

## Finite Time Lyapunov Exponents

(TODO)

## References

[1] Michael Dellnitz, Oliver Junge, and Gary Froyland. “The Algorithms Behind GAIO - Set Oriented Numerical Methods for Dynamical Systems”. In: _Ergodic Theory,Analysis, and Efficient Simulations of Dynamical Systems_. Ed. by Bernold Fiedler.Springer Berlin, 2001, pp. 145–174. doi: https://doi.org/10.1007/3-540-35593-6. 
