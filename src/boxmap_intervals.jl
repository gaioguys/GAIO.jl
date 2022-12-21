"""
    BoxMap(:interval, map, domain; no_subintervals::NTuple{N} = ntuple(_->4, N)) -> IntervalBoxMap
    BoxMap(:interval, map, domain; no_subintervals::Function) -> IntervalBoxMap

Type representing a discretization of a map using 
interval arithmetic to construct rigorous outer coverings 
of map images. `no_subintervals` describes how many times 
a given box will be subdivided before mapping. 
`no_subintervals` is a Function which 
has the signature `no_subintervals(center, radius)` and 
returns a tuple. If a tuple is passed directly for 
`no_subintervals`, then this is converted to a constant
Function `(_, _) -> no_subintervals`

Fields:
* `map`:              Map that defines the dynamical system.
* `domain`:           Domain of the map, `B`.
* `no_subintervals`:  Function with the signature 
                      `no_subintervals(center, radius)` which 
                      returns a tuple describing how many 
                      times a box is subdivided in each 
                      dimension before mapping. 

.
"""
struct IntervalBoxMap{N,T,I,F} <: BoxMap
    map::F
    domain::Box{N,T}
    no_subintervals::I

    function IntervalBoxMap(map::F, domain::Box{N,T}, no_subintervals::I) where {N,T,I,F}
        new{N,T,I,F}(map, domain, no_subintervals)
    end
    function IntervalBoxMap(map::F, domain::Box{N,T}, no_subintervals::I) where {N,T,I<:NTuple{N},F}
        new_no_subintervals = (c, r) -> no_subintervals
        new{N,T,typeof(new_no_subintervals),F}(map, domain, new_no_subintervals)
    end
end

function map_boxes(g::IntervalBoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
    P = source.partition
    @floop for box in source
        c, r = box
        int = IntervalBox(c .± r ...)
        for subint in mince(int, g.no_subintervals(c, r))
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            @reduce(image = BoxSet(P, S()) ⊔ P[fbox])
        end
    end
    return image
end

function construct_transfers(
        g::IntervalBoxMap, source::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q<:BoxPartition,S<:OrderedSet}

    P, D = source.partition, Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in source.set
        c, r = key_to_box(P, key)
        int = IntervalBox(c .± r ...)
        for subint in mince(int, g.no_subintervals(c, r))
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            Pbox = P[fbox]
            for hit in Pbox.set
                hit in source.set || @reduce( variant_keys = S() ⊔ hit )
                @reduce( mat = D() ⊔ ((hit,key) => 1) )
            end
        end
    end
    return mat, variant_keys
end

function IntervalBoxMap(map::F, domain::Box{N,T}; no_subintervals=ntuple(_->4,N)) where {N,T,F}
    IntervalBoxMap(map, domain, no_subintervals)
end

function IntervalBoxMap(map::F, P::BoxPartition{N,T}; no_subintervals=ntuple(_->4,N)) where {N,T,F}
    IntervalBoxMap(map, P.domain; no_subintervals=no_subintervals)
end

function Base.show(io::IO, g::IntervalBoxMap)
    center, radius = g.domain
    n = g.no_subintervals(center, radius)
    print(io, "IntervalBoxMap with $(n) subintervals")
end
