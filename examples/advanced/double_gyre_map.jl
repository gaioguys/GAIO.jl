@inline function rk4(f, x, τ)
    τ½ = τ / 2

    k = f(x)
    dx = @. k / 6

    k = f(@. x + τ½ * k)
    dx = @. dx + k / 3

    k = f(@. x + τ½ * k)
    dx = @. dx + k / 3

    k = f(@. x + τ * k)
    dx = @. dx + k / 6

    return @. x + τ * dx
end

function double_gyre(x, y, t; A = 0.25, ϵ = 0.25, ω = 2π)
    f(x, t) = ϵ * sin.(ω * t) .* x .^ 2 .+ (1 .- 2ϵ * (sin.(ω * t))) .* x
    df(x, t) = 2 * ϵ * sin.(ω * t) .* x .+ (1 .- 2ϵ * (sin.(ω * t)))

    return -π * A * sin.(π * f(x, t)) .* cos.(π * y),
    π * A * cos.(π * f(x, t)) .* sin.(π * y) .* df(x, t),
    1
end

function double_gyre_map(
    x;
    steps = 40,
    h = 0.25,
    return_intermediate = false,
    A = 0.25,
    ϵ = 0.25,
    ω = 2π,
    invert_time = false,
)
    factor = invert_time ? -1 : 1
    f = λ -> factor .* double_gyre(λ[1], λ[2], λ[3], A = A, ϵ = ϵ, ω = ω)
    if return_intermediate
        values = zeros(Float64, (steps, 3))
    end
    for i = 1:steps
        x = rk4(f, x, h)
        if return_intermediate
            values[i, :] = x
        end
    end
    if return_intermediate
        return values
    else
        return x
    end
end
