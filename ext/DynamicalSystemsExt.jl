module DynamicalSystemsExt

using GAIO, DynamicalSystemsBase, Base.Threads

import GAIO: preprocess
import Base: getindex

struct ParallelSystems{N,F}
    systems::NTuple{N,F}
end

function ParallelSystems(system::DynamicalSystem)
    # we are forced to use the interactive threads since FLoops doesn't 
    # know how to distinguish interactive threads from normal ones
    N = nthreads(:default) + nthreads(:interactive)
    systems = ntuple(_-> recursivecopy(system), N)
    return ParallelSystems(systems)
end

function (par::ParallelSystems{N,F})(x) where {N,F<:DiscreteTimeDynamicalSystem}
    system = par[threadid()]
    reinit!(system, x)
    step!(system)
    @debug "state" system maxlog=16
    return current_state(system)
end

function (par::ParallelSystems{N,F})(x) where {N,F<:ContinuousTimeDynamicalSystem}
    system = par[threadid()]
    reinit!(system, x)
    step!(system, 1, true)
    @debug "state" system maxlog=16
    return current_state(system)
end

getindex(par::ParallelSystems, i, j...) = getindex(par.systems, i, j...)

function preprocess(system::DynamicalSystem, args...)
    return ParallelSystems(system), args...
end

end # module
