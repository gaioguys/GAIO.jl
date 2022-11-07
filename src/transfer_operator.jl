mutable struct TransferOperator{B,T,S<:BoxSet{B},M<:BoxMap} <: AbstractSparseMatrix{T,Int}
    F::M
    support::S
    variant_set::S
    # it is more convenient to store the transposed matrix
    mat::SparseMatrixCSC{T,Int}
end

# helper function so we aren't doing type piracy on `mergewith!`
⊔(d::AbstractDict...) = mergewith!(+, d...)
⊔(d::AbstractDict, p::Pair...) = foreach(q -> d ⊔ q, p)
function ⊔(d::AbstractDict, p::Pair)
    k, v = p
    d[k] = haskey(d, k) ? d[k] + v : v
    d
end

# ensure that `TransferOperator` uses an `OrderedSet`
function TransferOperator(g::BoxMap, boxset::BoxSet{B,Q,S}) where {B,Q,S}
    TransferOperator(g, BoxSet(boxset.partition, OrderedSet(boxset.set)))
end

function TransferOperator(
        g::BoxMap, boxset::BoxSet{R,Q,S}
    ) where {N,T,R<:Box{N,T},Q<:BoxPartition,S<:OrderedSet}

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
                hit in boxset || @reduce( out_of_bounds = S() ⊔ hit )
                @reduce( dict = D() ⊔ ((key,hit) => inv_n) )
            end
        end
    end
    variant_set = BoxSet(P, out_of_bounds)
    mat = sparse(dict, boxset, variant_set)
    return TransferOperator(g, boxset, variant_set, mat)
end

function Base.show(io::IO, g::TransferOperator)
    print(io, "TransferOperator with $(length(nonzeros(g.mat))) stored entries over $(g.support.partition)")
end

Base.show(io::IO, ::MIME"text/plain", g::TransferOperator) = show(io, g)
Base.:(==)(g1::TransferOperator, g2::TransferOperator) = g1.mat == g2.mat
Base.eltype(::Type{<:TransferOperator{B,T}}) where {B,T} = T
Base.keytype(::Type{<:TransferOperator{B,T,I}}) where {B,T,I} = I
Base.size(g::TransferOperator) = (length(g.support) + length(g.variant_set), length(g.support))
Base.Matrix(g::TransferOperator) = copy(g.mat')

function Base.axes(g::TransferOperator)
    v = collect(g.support)
    u = vcat(v, collect(g.variant_set))
    return (u, v)
end
Base.setdiff!

function Base.checkbounds(::Type{Bool}, g::TransferOperator, keys)
    all(x -> checkbounds(Bool, g.support.partition, x), keys) || return false
    diff = setdiff(keys, g.support.set)
    if !isempty(diff)
        union!(g.support.set, keys)
        g̃ = TransferOperator(g.F, g.support)
        g.mat = g̃.mat
        g.variant_set = g̃.variant_set
    end
    return true
end

Base.checkbounds(b::Type{Bool}, g::TransferOperator, key1, keys...) = checkbounds(b, g, tuple(key1, keys...))

function Base.getindex(g::TransferOperator{T,I}, u, v) where {T,I}
    checkbounds(Bool, g, u, v) || throw(BoundsError(g, (u,v)))
    i, j = getkeyindex(g.support, u), getkeyindex(g.support, v)
    return g.mat[j, i]
end

function Base.setindex!(g::TransferOperator, u...)
    @error "setindex! is deliberately not supported for TransferOperators. Use getindex to generate an index value. "
end

for (type, (gmap, ind1, ind2, func)) in Dict(
        TransferOperator                                    => (:(g),        :j, :i, identity),
        LinearAlgebra.Transpose{<:Any,<:TransferOperator}   => (:(g.parent), :i, :j, transpose),
        LinearAlgebra.Adjoint{<:Any,<:TransferOperator}     => (:(g.parent), :i, :j, adjoint)
    )

    @eval begin

        LinearAlgebra.issymmetric(g::$type) = issymmetric($gmap.mat)

        function eigenfunctions(g::$type, B=I; nev=1, ritzvec=true, droptol=sqrt(eps(eltype($gmap))), kwargs...)
            λ, ϕ, nconv = Arpack._eigs(g, B; nev=nev, ritzvec=true, kwargs...)
            b = [BoxFun($gmap.support, ϕ[:, i]) for i in 1:nev]
            return ritzvec ? (λ, b, nconv) : (λ, nconv)
        end

        Arpack.eigs(g::$type, B::UniformScaling=I; kwargs...) = eigenfunctions(g, B; kwargs...)
        Arpack.eigs(g::$type, B; kwargs...) = eigenfunctions(g, B; kwargs...)
    
        LinearAlgebra.mul!(y::AbstractVector, g::$type, x::AbstractVector) = mul!(y, func($gmap.mat), x)
        
        function LinearAlgebra.mul!(y::BoxFun, g::$type, x::BoxFun)
            checkbounds(Bool, g, x.support.set) || throw(BoundsError(g, x.support.set))
            union!(y.support.set, g.support.set)
            resize!(y.vals, length(y.support.set))
            mul!(y.vals, g, x.vals)
            return y
        end

        function Base.:(*)(g::$type, x::BoxFun)
            y = BoxFun(x.support, similar(x.vals))
            return mul!(y.vals, g, x.vals)
        end

    end
end

# convert DOK sparse matrix to CSC using indices from support / variant_set
function SparseArrays.sparse(
        dict::Dict{Tuple{I,I},T}, support::BoxSet, variant_set::BoxSet
    ) where {I,T}

    m = length(support)
    n = m + length(variant_set)
    xs, ys, ws = Int[], Int[], T[]

    sizehint!(xs, length(dict))
    sizehint!(ys, length(dict))
    sizehint!(ws, length(dict))

    for ((u, v), w) in dict
        x = getkeyindex(support, u)
        y = v in support ? getkeyindex(support, v) : m + getkeyindex(variant_set, v)
        push!(xs, x)
        push!(ys, y)
        push!(ws, w)
    end

    return sparse(xs, ys, ws, m, n)
end
