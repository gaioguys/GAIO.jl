# Runge-Kutta scheme of 4th order
const half, sixth, third = Float32.((1/2, 1/6, 1/3))

"""
    rk4(f, x, τ)

Compute one step with step size `τ` of the classic 
fourth order Runge-Kutta method. 
"""
@muladd @inline function rk4(f, x, τ)
    τ½ = τ * half

    k = f(x)
    dx = @. k * sixth

    k = f(@. x + τ½ * k)
    dx = @. dx + k * third

    k = f(@. x + τ½ * k)
    dx = @. dx + k * third

    k = f(@. x + τ * k)
    dx = @. dx + k * sixth

    return @. x + τ * dx
end

"""
    rk4_flow_map(f, x, step_size=0.01, steps=20)

Perform `steps` steps of the classic Runge-Kutta fourth order method,
with step size `step_size`. 
"""
@inline function rk4_flow_map(f, x, step_size=0.01f0, steps=20)
    for _ in 1:steps
        x = rk4(f, x, step_size)
    end
    return x
end
