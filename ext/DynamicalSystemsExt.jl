module DynamicalSystemsExt

using GAIO, DynamicalSystemsBase, Base.Threads

import GAIO: preprocess
import Base: getindex, iterate

struct LockedSystem{F}
    lock::ReentrantLock
    system::F
end

LockedSystem(system) = LockedSystem(ReentrantLock(), system)
convert(::Type{T}, system::S) where {T<:LockedSystem,S<:DynamicalSystem} = LockedSystem(system)

Base.iterate(ls::LockedSystem, i...) = (ls.lock, Val(:syst))
Base.iterate(ls::LockedSystem, ::Val{:syst}) = (ls.system, Val(:done))
Base.iterate(ls::LockedSystem, ::Val{:done}) = nothing

struct ParallelSystems{N,F}
    systems::NTuple{N,LockedSystem{F}}
end

function ParallelSystems(system::DynamicalSystem)
    # we are forced to use the interactive threads since FLoops doesn't 
    # know how to distinguish interactive threads from normal ones
    N = nthreads(:default) + nthreads(:interactive)
    systems = ntuple(_-> LockedSystem(recursivecopy(system)), N)
    return ParallelSystems(systems)
end

function (par::ParallelSystems{N,F})(x) where {N,F<:DiscreteTimeDynamicalSystem}
    lk, system = par[threadid()]
    lock(lk) do 
        reinit!(system, x)
        step!(system)
        @debug "state" system maxlog=16
    end
    return current_state(system)
end

function (par::ParallelSystems{N,F})(x) where {N,F<:ContinuousTimeDynamicalSystem}
    lk, system = par[threadid()]
    lock(lk) do
        reinit!(system, x)
        step!(system, 1, true)
        @debug "state" system maxlog=16
    end
    return current_state(system)
end

getindex(par::ParallelSystems, i, j...) = getindex(par.systems, i, j...)

function preprocess(system::DynamicalSystem, args...)
    return ParallelSystems(system), args...
end

end # module
