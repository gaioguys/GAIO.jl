function v(x, tol)
    A = 2
    B = 1
    L = 0.9
    h = 2
    r = 2
    p = 3
    q = 4
    T = 2*π*A

    γ = t -> (cos(t) - (L*A)/(A+B) * cos((A+B)/A * t),
                sin(t) - (L*A)/(A+B) * sin((A+B)/A * t),
                -sin(B/A * t) + h * (1-2*t/T)^p)

    γ_t = t -> (-sin(t) + (L*A)/A * sin((A+B)/A * t),
                cos(t) - (L*A)/A * cos((A+B)/A * t),
                -cos(B/A * t) * B/A - h * p * (1-2*t/T)^(p-1) * 2/T)
    
    d = (x, y) -> inv(sum(abs.(x .- y) .^ q))

    σ = x -> (x[1]/(1-x[3]), x[2]/(1-x[3]))


    σ_inv = x -> (2*x[1]/(1 + x[1]^2 + x[2]^2), 2*x[2]/(1 + x[1]^2 + x[2]^2), 1 - 2/(1 + x[1]^2 + x[2]^2))

    # approximate the integral with adaptive Gauss quadrature for given tolerance
    int, = quadgk(t -> begin 
        γt = γ(t)
        γdt = γ_t(t) 
        nγdt = norm(γdt)
        dγt = d(x, γt)
        σγ = σ(γdt ./ nγdt)
        val = nγdt .* dγt .* σγ
        return SVector(val[1], val[2], nγdt * dγt)
    end, 0, T, rtol=tol)

    nom = (int[1], int[2])
    denom1 = int[3]
    y = (r * x[1] / norm((x[1], x[2])), r * x[2] / norm((x[1], x[2])), x[3])
    denom2 = d(x, y)

    w = nom ./ (denom1 + denom2)

    return σ_inv(w)
end

# interpolate the function v on the mesh [a,b]^3 with n equidistant nodes
# for simplicity we use linear B-Splines, using the equidistant mesh
# the interpolant will only be defined on [a,b]^3, so we have to adjust it to the dynamical system
function interpolate_v(a, b, n, tol)
    val = Array{SVector{3,Float64}, 3}(undef, n, n, n)
    x = range(a, b, length = n)
    y = range(a, b, length = n)
    z = range(a, b, length = n)

    @Threads.threads for i in 1:n
        for j in 1:n
            for k in 1:n
                val[i,j,k] = SVector(v((x[i], y[j], z[k]), tol))
            end
        end
    end
    
    itp = interpolate(val, BSpline(Linear()))
    sitp = scale(itp, x, y, z)
    
    return t -> sitp(t[1], t[2], t[3])
end

# do n steps of Runge-Kutta 4 with stepsize h
function knotted_f(v::F, x) where F
    h = 0.1
    n = 15

    for i in 1:n
        x = rk4(v, x, h)
    end

    return x
end

function knotted_flow(depth)
    generate_points = n -> [
        [(x, -1.0, 1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0, 1.0) for x in LinRange(-1, 1, n)];
        [(x, -1.0, -1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0, -1.0) for x in LinRange(-1, 1, n)];
        
        [(-1.0, x, 1.0) for x in LinRange(-1, 1, n)];
        [(1.0, x, 1.0) for x in LinRange(-1, 1, n)];
        [(-1.0, x, -1.0) for x in LinRange(-1, 1, n)];
        [(1.0, x, -1.0) for x in LinRange(-1, 1, n)];
        
        [(-1.0, 1.0, x) for x in LinRange(-1, 1, n)];
        [(1.0, 1.0, x) for x in LinRange(-1, 1, n)];
        [(-1.0, -1.0, x) for x in LinRange(-1, 1, n)];
        [(1.0, -1.0, x) for x in LinRange(-1, 1, n)];
    ]

    f = interpolate_v(-3.5, 3.5, 30, 1e-12)

    g = PointDiscretizedMap(x -> knotted_f(f, x), generate_points(40))
    partition = RegularPartition(Box(SVector(0.0, 0.0, 0.0), SVector(2.0, 2.0, 2.0)))
    boxset = partition[:]

    return chain_recurrent_set(boxset, g, depth)
end

function knotted_flow_rga(depth)
    generate_points = n -> [
        [(x, -1.0, 1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0, 1.0) for x in LinRange(-1, 1, n)];
        [(x, -1.0, -1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0, -1.0) for x in LinRange(-1, 1, n)];
        
        [(-1.0, x, 1.0) for x in LinRange(-1, 1, n)];
        [(1.0, x, 1.0) for x in LinRange(-1, 1, n)];
        [(-1.0, x, -1.0) for x in LinRange(-1, 1, n)];
        [(1.0, x, -1.0) for x in LinRange(-1, 1, n)];
        
        [(-1.0, 1.0, x) for x in LinRange(-1, 1, n)];
        [(1.0, 1.0, x) for x in LinRange(-1, 1, n)];
        [(-1.0, -1.0, x) for x in LinRange(-1, 1, n)];
        [(1.0, -1.0, x) for x in LinRange(-1, 1, n)];
    ]

    f = interpolate_v(-4.5, 4.5, 40, 1e-12)

    g = PointDiscretizedMap(x -> knotted_f(f, x), generate_points(40))
    partition = RegularPartition(Box(SVector(0.0, 0.0, 0.0), SVector(3.0, 3.0, 3.0)))
    boxset = partition[:]

    return relative_attractor(boxset, g, depth)
end
