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

# Constructors

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

function typesafe_map(g::IntervalBoxMap{N,T}, x) where {N,T}
    SVector{N,Interval{T}}(g.map(x)...)
end

# BoxMap API

function map_boxes(
        g::IntervalBoxMap, source::BS
    ) where {B,Q,S,BS<:BoxSet{B,Q,S}}

    P = source.partition
    @floop for box in source
        c, r = box
        for subint in mince(box, g.n_subintervals(c, r))
            fint = typesafe_map(g, subint)
            fSet = cover(P, fint)
            @reduce() do (image = BoxSet(P, S()); fSet)     # Initialize `image = BoxSet(P, S())` empty boxset
                image = image ⊔ fSet                        # Add `fSet` to `image`
            end
        end
    end
    return image::BS
end

function construct_transfers(
        g::IntervalBoxMap, domain::BS
    ) where {N,T,R<:Box{N,T},Q,S,BS<:BoxSet{R,Q,S}}

    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P, key)
        c, r = box
        for subint in mince(box, g.n_subintervals(c, r))
            fint = typesafe_map(g, subint)
            fSet = cover(P, fint)
            @reduce() do (image = BoxSet(P, S()); fSet)     # Initialize `image = BoxSet(P, S())` empty boxset
                image = image ⊔ fSet                        # Add `fSet` to `image`
            end

            for hit in fSet.set
                hitbox = key_to_box(P, hit)
                weight = (hit,key) => volume(fint ∩ hitbox)
                @reduce() do (mat = D(); weight)            # Initialize dict-of-keys sparse matrix
                    mat = mat ⊔ weight                      # Add weight to mat[hit, key]
                end
            end

        end
    end
    return mat::D, image::BS
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q}, codomain::BoxSet{U,H}
    ) where {N,T,R<:Box{N,T},Q,U,H}

    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P1, key)
        c, r = box
        for subint in mince(box, g.n_subintervals(c, r))
            fint = typesafe_map(g, subint)
            fSet = cover(P2, fint)

            for hit in keys(fSet)
                hit in codomain.set || continue
                hitbox = key_to_box(P2, hit)
                weight = (hit,key) => volume(fint ∩ hitbox)
                @reduce() do (mat = D(); weight)            # Initialize dict-of-keys sparse matrix
                    mat = mat ⊔ weight                      # Add weight to mat[hit, key]
                end
            end

        end
    end
    return mat::D
end
