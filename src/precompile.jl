@info "GAIO is caching common functions..."

#PrecompileTools.verbose[] = true

PrecompileTools.@setup_workload begin

    # shift first argument by 1/4
    __f(x, tail...) = (x+0.25f0, tail...)
    _f(x) = __f(x...)

    PrecompileTools.@compile_workload begin

        for dim in 2:3, T in (Float32, Float64)

            _c = _r = ntuple(_->T(0.5), dim)
            _Q = Box{T}(_c, _r)
            _P = BoxPartition(_Q, ntuple(_->2, dim))
            
            _S = cover(_P, :)
            _S = cover(_P, _Q)
            _S = cover(_P, ntuple(_->T(0), dim))

            _P = subdivide(_P, 1)
            _S = subdivide(_S, 1)

            _F = BoxMap(_f, _Q)
            _C = _F(_S)
            _𝔽 = TransferOperator(_F, _S)
            _𝔽 = TransferOperator(_F, _S, _S)

            _F = BoxMap(:interval, _f, _Q, n_subintervals=ntuple(_->1, dim))
            _C = _F(_S)
            _𝔽 = TransferOperator(_F, _S)
            _𝔽 = TransferOperator(_F, _S, _S)

            _F = BoxMap(:montecarlo, _f, _Q, n_points=1)
            _C = _F(_S)
            _𝔽 = TransferOperator(_F, _S)
            _𝔽 = TransferOperator(_F, _S, _S)

            _F = BoxMap(:grid, _f, _Q, n_points=ntuple(_->1, dim))
            _C = _F(_S)
            _𝔽 = TransferOperator(_F, _S)
            _𝔽 = TransferOperator(_F, _S, _S)

            _μ = BoxMeasure(_S, T)
            _ν = _𝔽*_μ
            _ν = _𝔽'_μ

            _A = preimage(_F, _S, _S)
            _A = preimage(_F, _S)
            _A = symmetric_image(_F, _S)
            _A = unstable_set(_F, _S)

            _A = relative_attractor(_F, _S, steps=1)
            _A = chain_recurrent_set(_F, _S, steps=1)
            _A = maximal_forward_invariant_set(_F, _S, steps=1)
            _A = maximal_invariant_set(_F, _S, steps=1)
            _σ = finite_time_lyapunov_exponents(_F, _S; T=T(1))

        end

    end

end

@info "GAIO is finished caching common functions."
