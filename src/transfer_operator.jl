struct TransferOperator{B,T,I,S<:BoxSet{B},M<:BoxMap,D<:AbstractDict{Tuple{I,I},T}} <: AbstractMatrix{T}
    F::M
    support::S
    mat::D
end

function TransferOperator(
        g::M, support::S, mat::D
    ) where {B,T,I,Q,Y<:OrderedSet,S<:BoxSet{B,Q,Y},M<:BoxMap,D<:AbstractDict{Tuple{I,I},T}}
    TransferOperator{B,T,I,S,M,D}(g, support, mat)
end

function TransferOperator(
        g::M, support::S, mat::D
    ) where {B,T,I,Q,Y,S<:BoxSet{B,Q,Y},M<:BoxMap,D<:AbstractDict{Tuple{I,I},T}}
    TransferOperator{B,T,I,S,M,D}(g, BoxSet(support.partition, OrderedSet(support.set)), mat)
end

# helper function so we aren't doing type piracy on `mergewith!`
⊔(d::AbstractDict...) = mergewith!(+, d...)
⊔(d::AbstractDict, p::Pair...) = foreach(q -> d ⊔ q, p)
function ⊔(d::AbstractDict, p::Pair)
    k, v = p
    d[k] = haskey(d, k) ? d[k] + v : v
    d
end

function TransferOperator(
        g::BoxMap, boxset::BoxSet{B,Q,S}
    ) where {N,T,B<:Box{N,T},Q<:BoxPartition,S}

    P, D = boxset.partition, Dict{Tuple{keytype(Q),keytype(Q)},T}
    @floop for key in boxset.set
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        domain_points = g.domain_points(c, r)
        inv_n = 1. / length(domain_points)
        for p in domain_points
            c = g.map(p)
            hitbox = point_to_box(P, c)
            isnothing(hitbox) && continue
            r = hitbox.radius
            for ip in g.image_points(c, r)
                hit = point_to_key(P, ip)
                isnothing(hit) && continue
                @reduce( mat = D() ⊔ ((hit,key) => inv_n) )
            end
        end
    end
    return TransferOperator(g, boxset, mat)
end

function Base.show(io::IO, g::TransferOperator)
    print(io, "TransferOperator with $(length(g.mat)) stored entries over $(g.partition)")
end

Base.show(io::IO, ::MIME"text/plain", g::TransferOperator) = show(io, g)
Base.:(==)(g1::TransferOperator, g2::TransferOperator) = g1.mat == g2.mat
Base.size(g::TransferOperator) = ((is, js) = axes(g); (length(is), length(js)))
Base.eltype(::Type{<:TransferOperator{B,T}}) where {B,T} = T
Base.keytype(::Type{<:TransferOperator{B,T,I}}) where {B,T,I} = Tuple{I,I}

function Base.axes(g::TransferOperator{B,T,I}) where {B,T,I}
    is, js = Set{I}(), Set{I}()
    for ((i, j), w) in g.mat
        is = is ⊔ i
        js = js ⊔ j
    end
    return collect(is), collect(js)
end

function Base.checkbounds(::Type{Bool}, g::TransferOperator{B,T,I}, keys) where {B,T,I}
    all(x -> checkbounds(Bool, g.partition, x), keys) || return false
    diff = setdiff(keys, axes(g)[2])
    if !isempty(diff)
        g̃ = TransferOperator(g.F, BoxSet(g.partition, Set{I}(diff)))
        g.mat = g.mat ⊔ g̃.mat
    end
    return true
end

Base.checkbounds(b::Type{Bool}, g::TransferOperator, key1, keys...) = checkbounds(b, g, tuple(key1, keys...))

function Base.getindex(g::TransferOperator{T,I}, u, v) where {T,I}
    checkbounds(Bool, g, u, v) || throw(BoundsError(g, (u,v)))
    return g.mat[u, v]
end

function Base.setindex!(g::TransferOperator, u...)
    @error "setindex! is deliberately not supported for TransferOperators. Use getindex to generate an index value. "
end

for (type, (gmap, ind1, ind2, func)) in Dict(
        TransferOperator                                    => (:(g),        :i, :j, identity),
        LinearAlgebra.Transpose{<:Any,<:TransferOperator}   => (:(g.parent), :j, :i, transpose),
        LinearAlgebra.Adjoint{<:Any,<:TransferOperator}     => (:(g.parent), :j, :i, adjoint)
    )

    @eval begin

        function LinearAlgebra.issymmetric(g::$type)
            for ((i, j), w) in $gmap.mat
                w̃ = get($gmap.mat, (j,i), 0)
                w == w̃ || return false
            end
            return true
        end

        function eigenfunctions(g::$type, B=I; nev=1, ritzvec=true, droptol=sqrt(eps(eltype($gmap))), kwargs...)
            λ, ϕ, nconv = Arpack._eigs(g, B; nev=nev, ritzvec=true, kwargs...)
            D = OrderedDict{keytype($gmap.support.partition),eltype(ϕ)}
            b = [
                BoxFun(
                    $gmap.partition, 
                    D(key => val for (key,val) in zip(g.support.set, ϕ[:, i]))
                ) for i in 1:nev
            ]
            return ritzvec ? (λ, b, nconv) : (λ, nconv)
        end

        Arpack.eigs(g::$type, B::UniformScaling=I; kwargs...) = eigenfunctions(g, B; kwargs...)
        Arpack.eigs(g::$type, B; kwargs...) = eigenfunctions(g, B; kwargs...)
    
        @muladd function LinearAlgebra.mul!(y::AbstractVector, g::$type, x::AbstractVector)
            s = $gmap.support
            for ((u, v), w) in $gmap.mat
                (u in s.set && v in s.set) || continue
                $ind1, $ind2 = getkeyindex(s, u), getkeyindex(s, v)
                y[j] = y[j] + w * $func(x[i])
            end
            return y
        end

        @muladd function LinearAlgebra.mul!(y::BoxFun, g::$type, x::BoxFun)
            s = $gmap.support
            for ((u, v), w) in $gmap.mat
                $ind1, $ind2 = u, v
                y[j] = y[j] + w * $func(x[i])
            end
            return y
        end

        function Base.:(*)(g::$type, x::BoxFun{B,K,V,P,D}) where {B,K,V,P,D}
            checkbounds(Bool, $gmap, keys(x.vals)) || throw(BoundsError(g, x))
            vals = D()
            sizehint!(vals, length(x))
            y = BoxFun(x.partition, vals)
            return mul!(y.vals, g, x.vals)
        end

    end
end
