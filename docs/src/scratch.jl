function ω(F::BoxMap, S::BoxSet)
    Sₖ = S
    Sₖ₊₁ = empty(S)

    while Sₖ₊₁ ≠ Sₖ
        Sₖ = Sₖ₊₁
        Sₖ₊₁ = Sₖ ∩ F(Sₖ)
    end

    return Sₖ
end



function α(F::BoxMap, S::BoxSet)
    F⁻¹(B) = preimage(F, B, B)  # actually computes  F⁻¹(B) ∩ B  
    return ω(F⁻¹, S)            # which is all we need
end

"""
    iterate_until(f, start, stop_condition; max_iterations = Inf, initial_check = false)

Iterate a function `f` starting at `start` until 
some `stop_condition` is reached or `max_iterations` 
have been performed. 
One iteration is always guaranteed! 
"""
function iterate_until(f, start, stop_condition; max_iterations=Inf)
    iter = f(start)
    n_iterations = 1

    while !stop_condition(iter)  &&  n_iterations < max_iterations
        iter = f(iter)
        n_iterations += 1
    end
    
    @debug "number of iterations" n_iterations max_iterations
    return iter
end

"""
    iterate_until_equal(f, start; max_iterations = Inf)

Iterate a function `f` starting at `start` until 
a fixed point is reached or `max_iterations` 
have been performed. 
One iteration is always guaranteed! 
"""
function iterate_until_equal(f, start; max_iterations=Inf)
    new_start = (start, start)
    new_f((x, x_old)) = (f(x), x)
    stop_condition((x, x_old)) = x == x_old

    return iterate_until(new_f, new_start, stop_condition; max_iterations=max_iterations) 
end