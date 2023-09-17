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
struct IntervalBoxMap{N,T} <: BoxMap
    map
    domain::Box{N,T}
    n_subintervals

    function IntervalBoxMap(map, domain::Box{N,T}, n_subintervals) where {N,T}
        new{N,T}(map, domain, n_subintervals)
    end
    function IntervalBoxMap(map, domain::Box{N,T}, n_subintervals::NTuple{N}) where {N,T}
        new_n_subintervals = (c, r) -> n_subintervals
        new{N,T}(map, domain, new_n_subintervals)
    end
end

function map_boxes(
        g::IntervalBoxMap, source::BS;
        show_progress=false
    ) where {B,Q,S,BS<:BoxSet{B,Q,S}}

    prog = Progress(length(source)+1; desc="Computing image...", enabled=show_progress, showspeed=true)

    P = source.partition
    @floop for box in source
        c, r = box
        int = IntervalBox(box)
        minced = mince(int, g.n_subintervals(c, r))
        show_progress && @reduce( n_ints_mapped = 0 + length(minced) )
        for subint in minced
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P, fbox)
            @reduce( image = BoxSet(P, S()) ⊔ fSet )
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    return image::BS
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q,S};
        show_progress=false
    ) where {N,T,R<:Box{N,T},Q,S}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", enabled=show_progress, showspeed=true)

    P, D = domain.partition, Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in domain.set
        box = key_to_box(P, key)
        c, r = box
        int = IntervalBox(box)
        minced = mince(int, g.n_subintervals(c, r))
        show_progress && @reduce( n_ints_mapped = 0 + length(minced) )
        for subint in minced
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P, fbox)
            for hit in fSet.set
                @reduce( image = S() ⊔ hit )
                hitbox = fbox ∩ key_to_box(P, hit)
                isnothing(hitbox) && continue
                V = volume(hitbox)
                @reduce( mat = D() ⊔ ((hit,key) => V) )
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    image_set = BoxSet(P, image::S)
    return mat::D, image_set
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q,S}, codomain::BoxSet{U,H,W};
        show_progress=false
    ) where {N,T,R<:Box{N,T},Q,S,U,H,W}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", enabled=show_progress, showspeed=true)

    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}
    @floop for key in domain.set
        box = key_to_box(P1, key)
        c, r = box
        int = IntervalBox(box)
        minced = mince(int, g.n_subintervals(c, r))
        show_progress && @reduce( n_ints_mapped = 0 + length(minced) )
        for subint in minced
            fint = g.map(subint)
            fbox = Box(fint)
            isnothing(fbox) && continue
            fSet = cover(P2, fbox)
            for hit in fSet.set
                hit in codomain.set || continue
                hitbox = fbox ∩ key_to_box(P2, hit)
                isnothing(hitbox) && continue
                V = volume(hitbox)
                @reduce( mat = D() ⊔ ((hit,key) => V) )
            end
        end
        next!(prog)
    end
    
    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    return mat::D
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
