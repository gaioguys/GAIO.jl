module ProgressMeterExt

using GAIO, ProgressMeter, IntervalArithmetic, StaticArrays, FLoops#, SIMD

import GAIO: typesafe_map, map_boxes, construct_transfers, ⊔, SVNT

# IntervalBoxMap

function map_boxes(
        g::IntervalBoxMap, source::BS,
        show_progress::Val{true}
    ) where {B,Q,S,BS<:BoxSet{B,Q,S}}

    prog = Progress(length(source)+1; desc="Computing image...", showspeed=true)

    P = source.partition
    @floop for box in source
        c, r = box
        minced = mince(box, g.n_subintervals(c, r))

        n = length(minced)
        @reduce( n_ints_mapped += n )

        for subint in minced
            fint = g.map(subint)
            fSet = cover(P, fint)

            @reduce() do (image = BoxSet(P, S()); fSet)     # Initialize `image = BoxSet(P, S())` empty boxset
                image = image ⊔ fSet                        # Add `fSet` to `image`
            end
        
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    return image::BS
end

function construct_transfers(
        g::IntervalBoxMap, domain::BS,
        show_progress::Val{true}
    ) where {N,T,R<:Box{N,T},Q,S,BS<:BoxSet{R,Q,S}}

    prog = Progress(length(source)+1; desc="Computing image...", showspeed=true)
    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P, key)
        c, r = box
        minced = mince(box, g.n_subintervals(c, r))

        n = length(minces)
        @reduce( n_ints_mapped += n )

        for subint in minced
            fint = g.map(subint)
            fSet = cover(P, fint)

            @reduce() do (image = BoxSet(P, S()); fSet)     # Initialize `image = BoxSet(P, S())` empty boxset
                image = image ⊔ fSet                        # Add `fSet` to `image`
            end

            for hit in keys(fSet)
                hitbox = key_to_box(P, hit)
                weight = (hit,key) => volume(fint ∩ hitbox)
                @reduce() do (mat = D(); weight)            # Initialize dict-of-keys sparse matrix
                    mat = mat ⊔ weight                      # Add weight to mat[hit, key]
                end
            end

        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    return mat::D, image::BS
end

function construct_transfers(
        g::IntervalBoxMap, domain::BoxSet{R,Q}, codomain::BoxSet{U,H},
        show_progress::Val{true}
    ) where {N,T,R<:Box{N,T},Q,U,H}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", showspeed=true)
    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P1, key)
        c, r = box
        minced = mince(box, g.n_subintervals(c, r))

        n = length(minced)
        @reduce( n_ints_mapped += n )

        for subint in minced
            fint = g.map(subint)
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
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped intervals", n_ints_mapped)])
    return mat::D
end

# SampledBoxMap

function map_boxes(
        g::SampledBoxMap, source::BoxSet{B,Q,S}, 
        show_progress::Val{true}
    ) where {B,Q,S}

    prog = Progress(length(source)+1; desc="Computing image...", showspeed=true)

    P = source.partition
    @floop for box in source
        c, r = box
        domain_points = g.domain_points(c, r)

        n = length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            c = typesafe_map(g, p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                @reduce() do (image = S(); hit)     # Initialize empty key set
                    image = image ⊔ hit             # Add hit key to image
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    return BoxSet(P, image::S)
end 

function construct_transfers(
        g::SampledBoxMap, domain::BoxSet{R,Q,S},
        show_progress::Val{true}
    ) where {N,T,R<:Box{N,T},Q,S}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", showspeed=true)
    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P, key)
        c, r = box
        domain_points = g.domain_points(c, r)

        n = length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            c = typesafe_map(g, p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                weight = (hit,key) => 1
                @reduce() do (image = S(); hit), (mat = D(); weight)    # Initialize empty key set and dict-of-keys sparse matrix
                    image = image ⊔ hit                                 # Add hit key to image
                    mat = mat ⊔ weight                                  # Add weight to mat[hit, key]
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    codomain = BoxSet(P, image::S)
    return mat::D, codomain
end

function construct_transfers(
        g::SampledBoxMap, domain::BoxSet{R,Q}, codomain::BoxSet{U,H},
        show_progress::Val{true}
    ) where {N,T,R<:Box{N,T},Q,U,H}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", showspeed=true)
    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}

    @floop for key in keys(domain)
        box = key_to_box(P1, key)
        c, r = box
        domain_points = g.domain_points(c, r)
        
        n = length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            c = typesafe_map(g, p)
            hitbox = point_to_box(P2, c)
            isnothing(hitbox) && continue
            _, r = hitbox
            for ip in g.image_points(c, r)
                hit = point_to_key(P2, ip)
                hit in codomain.set || continue
                weight = (hit,key) => 1
                @reduce() do (mat = D(); weight)     # Initialize dict-of-keys sparse matrix
                    mat = mat ⊔ weight               # Add weight to mat[hit, key]
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    return mat::D
end

# CPUSampledBoxMap
# Julia currently does not support nested package extensions,
# the following is a hack that only worked in a beta version of 1.9. 
# For now, CPUSampledBoxMap and GPUSampledBoxMap are not supported with progress meters

#=
SIMDExt = Base.get_extension(GAIO, :SIMDExt)
import SIMDExt: CPUSampledBoxMap, tuple_vgather, tuple_vscatter!

@inbounds function map_boxes(
        G::CPUSampledBoxMap{simd,N,T}, source::BoxSet{B,Q,S},
        show_progress::Val{true}
    ) where {simd,N,T,B,Q,S}

    prog = Progress(length(source)+1; desc="Computing image...", showspeed=true)
    P = source.partition
    g, idx_base = G

    @floop for box in source
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        c, r = box
        domain_points = g.domain_points(c, r)
        
        n = simd * length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P, ip)
                    @reduce() do (image = S(); hit)     # Initialize empty key set
                        image = image ⊔ hit             # Add hit key to image
                    end
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    return BoxSet(P, image::S)
end

@inbounds function construct_transfers(
        G::CPUSampledBoxMap{simd}, domain::BoxSet{R,Q,S},
        show_progress::Val{true}
    ) where {simd,N,T,R<:Box{N,T},Q,S}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", showspeed=true)
    P = domain.partition
    D = Dict{Tuple{keytype(Q),keytype(Q)},T}
    g, idx_base = G

    @floop for key in keys(domain)
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        box = key_to_box(P, key)
        c, r = box
        domain_points = g.domain_points(c, r)
        
        n = simd * length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P, ip)
                    weight = (hit,key) => one(T)
                    @reduce() do (image = S(); hit), (mat = D(); weight)    # Initialize empty key set and dict-of-keys sparse matrix
                        image = image ⊔ hit                                 # Add hit key to image
                        mat = mat ⊔ weight                                  # Add weight to mat[hit, key]
                    end
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    codomain = BoxSet(P, image::S)
    return mat::D, codomain
end

@inbounds @muladd function construct_transfers(
        G::CPUSampledBoxMap{simd}, domain::BoxSet{R,Q}, codomain::BoxSet{U,H},
        show_progress::Val{true}
    ) where {simd,N,T,R<:Box{N,T},Q,U,H}

    prog = Progress(length(domain)+1; desc="Computing transfer weights...", showspeed=true)
    P1, P2 = domain.partition, codomain.partition
    D = Dict{Tuple{keytype(H),keytype(Q)},T}
    g, idx_base = G

    @floop for key in keys(domain)
        @init mapped_points = Vector{SVector{N,T}}(undef, simd)
        box = key_to_box(P1, key)
        c, r = box
        domain_points = g.domain_points(c, r)
        
        n = simd * length(domain_points)
        @reduce( n_points_mapped += n )

        for p in domain_points
            fp = typesafe_map(G, p)
            tuple_vscatter!(mapped_points, fp, idx_base)
            for q in mapped_points
                hitbox = point_to_box(P2, q)
                isnothing(hitbox) && continue
                _, r = hitbox
                for ip in g.image_points(q, r)
                    hit = point_to_key(P2, ip)
                    hit in codomain.set || continue
                    weight = (hit,key) => one(T)
                    @reduce() do (mat = D(); weight)     # Initialize dict-of-keys sparse matrix
                        mat = mat ⊔ weight               # Add weight to mat[hit, key]
                    end
                end
            end
        end
        next!(prog)
    end

    next!(prog, showvalues=[("Total number of mapped points", n_points_mapped)])
    return mat::D
end

# GPUSampledBoxMap
=#

end
