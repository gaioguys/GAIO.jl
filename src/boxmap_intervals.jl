"""
    BoxMap(:interval, map, domain::Box{N}) -> IntervalBoxMap
    BoxMap(:interval, map, domain::Box{N}) -> IntervalBoxMap

Type representing a discretization of a map using 
interval arithmetic to construct rigorous outer coverings 
of map images. 

Fields:
* `map`:              Map that defines the dynamical system.
* `domain`:           Domain of the map, `B`.

.
"""
struct IntervalBoxMap{N,T} <: BoxMap
    map
    domain::Box{N,T}
end

# Constructors

IntervalBoxMap(map, P::Q) where {Q<:AbstractBoxPartition} = IntervalBoxMap(map, P.domain)

Base.show(io::IO, ::IntervalBoxMap) = print(io, "IntervalBoxMap")

typesafe_map(g::IntervalBoxMap{N,T}, x) where {N,T} = Box{N,T}(g.map(x)...)

# BoxMap API

function map_boxes(
        g::IntervalBoxMap, source::BS
    ) where {B,Q,S,BS<:BoxSet{B,Q,S}}

    P = source.partition
    @floop for box in source
        c, r = box
        fint = typesafe_map(g, c .± r)
        fset = cover(P, fint)
        @reduce() do (image = BoxSet(P, S()); fset)     # Initialize `image = BoxSet(P, S())` empty boxset
            image = image ⊔ fset                        # Add `fSet` to `image`
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
        fint = typesafe_map(g, c .± r)
        fset = cover(P, fint)
        @reduce() do (image = BoxSet(P, S()); fset)     # Initialize `image = BoxSet(P, S())` empty boxset
            image = image ⊔ fset                        # Add `fSet` to `image`
        end

        for hit in keys(fset)
            hitbox = key_to_box(P, hit)
            weight = (hit,key) => volume(fint ∩ hitbox)
            @reduce() do (mat = D(); weight)            # Initialize dict-of-keys sparse matrix
                mat = mat ⊔ weight                      # Add weight to mat[hit, key]
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
        fint = typesafe_map(g, c .± r)
        fset = cover(P2, fint)

        for hit in keys(fset)
            hit in codomain.set || continue
            hitbox = key_to_box(P2, hit)
            weight = (hit,key) => volume(fint ∩ hitbox)
            @reduce() do (mat = D(); weight)            # Initialize dict-of-keys sparse matrix
                mat = mat ⊔ weight                      # Add weight to mat[hit, key]
            end
        end
    end
    return mat::D
end
