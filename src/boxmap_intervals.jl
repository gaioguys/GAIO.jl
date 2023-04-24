"""
    BoxMap(:interval, map, domain::Box{N}; n_subintervals::NTuple{N} = ntuple(_->4, N)) -> IntervalBoxMap
    BoxMap(:interval, map, domain::Box{N}; n_subintervals::Function) -> IntervalBoxMap

Type representing a discretization of a map using 
interval arithmetic to construct rigorous outer coverings 
of map images. `n_subintervals` describes how many times 
a given box will be subdivided before mapping. 
`n_subintervals` is a Function which 
has the signature `n_subintervals(center, radius)` and 
returns a tuple. If a tuple is passed directly for 
`n_subintervals`, then this is converted to a constant
Function `(_, _) -> n_subintervals`

Fields:
* `map`:              Map that defines the dynamical system.
* `domain`:           Domain of the map, `B`.
* `n_subintervals`:   Function with the signature 
                      `n_subintervals(center, radius)` which 
                      returns a tuple describing how many 
                      times a box is subdivided in each 
                      dimension before mapping. 

.
"""
struct IntervalBoxMap{N,T,I,F} <: BoxMap
    map::F
    domain::Box{N,T}
    n_subintervals::I

    function IntervalBoxMap(map::F, domain::Box{N,T}, n_subintervals::I) where {N,T,I,F}
        new{N,T,I,F}(map, domain, n_subintervals)
    end
    function IntervalBoxMap(map::F, domain::Box{N,T}, n_subintervals::I) where {N,T,I<:NTuple{N},F}
        new_n_subintervals = (c, r) -> n_subintervals
        new{N,T,typeof(new_n_subintervals),F}(map, domain, new_n_subintervals)
    end
end

function map_boxes(g::IntervalBoxMap, source::BoxSet{B,Q,S}) where {B,Q,S}
    P = source.partition
    @floop for box in source
        c, r = box
        int = IntervalBox(box)
        for subint in mince(int, g.n_subintervals(c, r))
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P, fbox)
            @reduce(image = BoxSet(P, S()) ⊔ fSet)
        end
    end
    return image
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q,S}

    P, D = domain.partition, Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in domain.set
        box = key_to_box(P, key)
        c, r = box
        int = IntervalBox(box)
        for subint in mince(int, g.n_subintervals(c, r))
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P, fbox)
            for hit in fSet.set
                @reduce( image = S() ⊔ hit )
                hitbox = key_to_box(P, hit)
                V = volume(fbox ∩ hitbox)
                @reduce( mat = D() ⊔ ((hit,key) => V) )
            end
        end
    end
    image_set = BoxSet(P, image)
    return mat, image_set
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q,S}, codomain::BoxSet{U,H,W}
    ) where {N,T,R<:Box{N,T},Q,S,U,H,W}

    P1, P2 = domain.partiiton, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}
    @floop for key in domain.set
        box = key_to_box(P1, key)
        c, r = box
        int = IntervalBox(box)
        for subint in mince(int, g.n_subintervals(c, r))
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P2, fbox)
            for hit in fSet.set
                hit in codomain.set || continue
                hitbox = key_to_box(P, hit)
                V = volume(fbox ∩ hitbox)
                @reduce( mat = D() ⊔ ((hit,key) => V) )
            end
        end
    end
    return mat
end

function IntervalBoxMap(map, domain::Box{N,T}; n_subintervals=ntuple(_->4,N)) where {N,T}
    IntervalBoxMap(map, domain, n_subintervals)
end

function IntervalBoxMap(map, P::Q; n_subintervals=ntuple(_->4,N)) where {N,T,Q<:AbstractBoxPartition{Box{N,T}}}
    IntervalBoxMap(map, P.domain; n_subintervals=n_subintervals)
end

function Base.show(io::IO, g::IntervalBoxMap)
    n = g.n_subintervals(g.domain...)
    print(io, "IntervalBoxMap with $(n) subintervals")
end
