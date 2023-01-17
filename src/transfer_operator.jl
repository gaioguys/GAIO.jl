"""
    TransferOperator(map::BoxMap, support::BoxSet)

Discretization of the Perron-Frobenius operator, or transfer operator. 
Implemented as a sparse matrix with the same linear indices as `support`,
e.g. if 
```julia
julia> B = BoxSet(partition, [3,10,30])
  Boxset over [...] partition

julia> T = TransferOperator(boxmap, B)
  TransferOperator over [...] BoxSet
```
for some `partition` and `boxmap`, then 
```julia
julia> axes(T)
  ([3, 10, 30], [3, 10, 30])
```

Fields:
* `boxmap`:         `SampledBoxMap` map which relates to the transfers.
* `support`:        `BoxSet` which contains keys for the already calculated transfers. 
                    Effectively, these are row/column pointers, i.e. the 
                    first column of `T.mat` contains transfer weights FROM 
                    box B_1, where B_1 is the first box of `support`. 
* `variant_set`:    `BoxSet` which contains keys for potential boxes lying 
                    outside of `support`, i.e. it could be that `support` is 
                    not an invariant set. In this case, `variant_set` contains 
                    the boxes which were not in `support`, but whose preimage 
                    lies in `support`. 
* `mat`:            `SparseMatrixCSC` containing transfer weights. The index 
                    `T.mat[i,j]` represents the transfer weight FROM the `j`'th
                    box in `support` TO the `i`'th box in `support`. If `support` 
                    is not invariant, then the matrix will be tall. In this case 
                    the rows are counted in `support` and then in `variant_set`. 

```julia
support -->
  |     .   .   .   .   .
  |     .   .   .   .   .
  v     .   .   .   .   .
        .   .  mat  .   .
        .   .   .   .   .
 var-   .   .   .   .   .
 ian-   .   .   .   .   .
 t_set  .   .   .   .   .     
  |     .   .   .   .   .
  |     .   .   .   .   .
  v     .   .   .   .   .
```

It is important to note that `TranferOperator` is only supported over the 
box set `B`, but if one lets a `TranferOperator` act on a `BoxFun`, then 
the support `B` is extended "on the fly" to include the support of the `BoxFun`.

Methods Implemented: 
```julia
:(==), axes, size, eltype, getindex, setindex!, SparseArrays.sparse, Arpack.eigs, LinearAlgebra.mul! #, etc ...
```

Implementation detail:

The reader may have noticed that the matrix representation 
depends on the order of boxes in `support`. For this reason 
an `OrderedSet` is used. `BoxSet`s using regular `Set`s 
will be copied and converted to `OrderedSet`s. 

.
"""
mutable struct TransferOperator{B,T,S<:BoxSet{B},M<:BoxMap} <: AbstractSparseMatrix{T,Int}
    boxmap::M
    domain::S
    codomain::S
    mat::SparseMatrixCSC{T,Int}
end

construct_transfers(g::TransferOperator, domain::BoxSet) = construct_transfers(g.boxmap, domain)

# ensure that `TransferOperator` uses an `OrderedSet`
function TransferOperator(g::BoxMap, domain::BoxSet{B,P,S}, codomain::BoxSet{R,Q,W}) where {B,P,S,R,Q,W}
    dom = BoxSet(domain.partition, OrderedSet(domain.set))
    codom = BoxSet(codomain.partition, OrderedSet(codomain.set))
    TransferOperator(g, dom, codom)
end

function TransferOperator(
        g::BoxMap, domain::BoxSet{B,P,S}, codomain::BoxSet{R,Q,W}
    ) where {B,P,S<:OrderedSet,R,Q,W<:OrderedSet}

    dict = construct_transfers(g, domain, codomain)
    mat = sparse(dict, domain, codomain)

    for i in 1:size(mat, 2)
        mat[:, i] ./= sum(mat[:, i])
    end

    return TransferOperator(g, boxset, variant_set, mat)
end

function Base.show(io::IO, g::TransferOperator)
    print(io, "TransferOperator over $(g.support)")
end

function Base.show(io::IO, ::MIME"text/plain", g::TransferOperator)
    print(io, "TransferOperator over $(g.support) with $(length(nonzeros(g.mat))) stored entries:\n\n")
    SparseArrays._show_with_braille_patterns(io, g.mat)
end

Base.:(==)(g1::TransferOperator, g2::TransferOperator) = g1.mat == g2.mat
Base.eltype(::Type{<:TransferOperator{B,T}}) where {B,T} = T
Base.keytype(::Type{<:TransferOperator{B,T,I}}) where {B,T,I} = I
Base.size(g::TransferOperator) = size(g.mat)
Base.Matrix(g::TransferOperator) = Matrix(g.mat)
SparseArrays.sparse(g::TransferOperator) = copy(g.mat)

function Base.axes(g::TransferOperator)
    v = collect(g.support)
    u = vcat(v, collect(g.variant_set))
    return (u, v)
end

function Base.getindex(g::TransferOperator{T,I}, u, v) where {T,I}
    checkbounds(Bool, g, u, v) || throw(BoundsError(g, (u,v)))
    i, j = getkeyindex(g.support, u), getkeyindex(g.support, v)
    return g.mat[i, j]
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

        LinearAlgebra.issymmetric(g::$type) = issymmetric($gmap.mat)

        function eigenfunctions(g::$type, B=I; nev=1, ritzvec=true, kwargs...)
            λ, ϕ, nconv = Arpack._eigs(g, B; nev=nev, ritzvec=true, kwargs...)
            S = $gmap.support
            P = S.partition
            b = [BoxFun(S, ϕ[:, i], OrderedDict) for i in 1:nev]
            return ritzvec ? (λ, b, nconv) : (λ, nconv)
        end

        """
            eigs(gstar::TransferOperator [; kwargs...]) -> (d[, v], nconv)

        Compute a set of eigenvalues `d` and eigenmeasures `v` of `gstar`. 
        Works with the adjoint _Koopman operator_ as well. 
        All keyword arguments from `Arpack.eigs` can be passed. See the 
        documentation for `Arpack.eigs`. 
        """
        Arpack.eigs(g::$type, B::UniformScaling=I; kwargs...) = eigenfunctions(g, B; kwargs...)
        Arpack.eigs(g::$type, B; kwargs...) = eigenfunctions(g, B; kwargs...)

        LinearAlgebra.mul!(y::AbstractVector, g::$type, x::AbstractVector) = mul!(y, $func($gmap.mat), x)

        @inline function LinearAlgebra.mul!(y::BoxFun, g::$type, x::BoxFun)
            @boundscheck(checkbounds(Bool, g, keys(x.vals)) || throw(BoundsError(g, x)))
            zer = zero(eltype(y))
            map!(x -> zer, values(y.vals))
            rows = rowvals($gmap.mat)
            vals = nonzeros($gmap.mat)
            # iterate over columns with the keys that the columns represent
            for (col_j, j) in enumerate($gmap.support.set)
                for k in nzrange($gmap.mat, col_j)
                    w = vals[k]
                    row_i = rows[k]
                    # grab the key that this row represents
                    i = getindex_fromkeys($gmap, row_i)
                    y[$ind1] = @muladd y[$ind1] + $func(w) * x[$ind2]
                end
            end
            return y
        end

        function Base.:(*)(g::$type, x::BoxFun{B,K,V}) where {B,K,V}
            dict = OrderedDict{K, promote_type(eltype(g), V)}()
            sizehint!(dict, length($gmap.support) + length($gmap.variant_set))
            y = BoxFun($gmap.support.partition, dict)
            return mul!(y, g, x)
        end

    end
end

# helper function to force rehash `Set` / `OrderdSet` objects
_rehash!(dict::Dict) = Base.rehash!(dict)
_rehash!(dict::OrderedDict) = OrderedCollections.rehash!(dict)
_rehash!(dict::IdDict) = Base.rehash!(dict)
_rehash!(set::Union{Set,OrderedSet}) = _rehash!(set.dict)
_rehash!(d::Union{AbstractDict,AbstractSet}) = d
_rehash!(boxset::BoxSet) = _rehash!(boxset.set)

# helper function to access `Set` / `OrderedSet` internals
getkeyindex(dict::Dict, i) = (j = Base.ht_keyindex(dict, i); j > 0 ? j : nothing)
getkeyindex(set::Set, i) = (j = Base.ht_keyindex(set.dict, i); j > 0 ? j : nothing)
getkeyindex(dict::OrderedDict, i) = (j = OrderedCollections.ht_keyindex(dict, i, true); j > 0 ? j : nothing)
getkeyindex(set::OrderedSet, i) = (j = OrderedCollections.ht_keyindex(set.dict, i, true); j > 0 ? j : nothing)
getkeyindex(boxset::BoxSet, i) = getkeyindex(boxset.set, i)

getindex_fromkeys(dict::Union{Dict,OrderedDict}, i) = dict.keys[i]
getindex_fromkeys(set::Union{Set,OrderedSet}, i) = getindex_fromkeys(set.dict, i)
getindex_fromkeys(boxset::BoxSet, i) = getindex_fromkeys(boxset.set, i)
getindex_fromkeys(g::TransferOperator, i, j) = (getindex_fromkeys(g, i), getindex_fromkeys(g, j))

@inline function getindex_fromkeys(g::TransferOperator, i)
    m, n = size(g)
    @boundscheck checkbounds(Base.OneTo{Int}(m), i)
    if i ≤ n 
        k = getindex_fromkeys(g.support, i)
    else
        k = getindex_fromkeys(g.variant_set, i - n) 
    end
    return k
end

# convert DOK sparse matrix to CSC using indices from support / variant_set
function SparseArrays.sparse(
        dict::Dict{Tuple{I,I},T}, support::BoxSet, variant_set::BoxSet
    ) where {I,T}

    _rehash!(support)
    _rehash!(variant_set)

    n = length(support)
    m = n + length(variant_set)
    xs, ys, ws = Int[], Int[], T[]

    sizehint!(xs, length(dict))
    sizehint!(ys, length(dict))
    sizehint!(ws, length(dict))

    for ((u, v), w) in dict
        x = u in support.set ? getkeyindex(support, u) : n + getkeyindex(variant_set, u)
        y = getkeyindex(support, v)
        push!(xs, x)
        push!(ys, y)
        push!(ws, w)
    end

    return sparse(xs, ys, ws, m, n)
end

function Base.Dict(g::TransferOperator{B,T,S}) where {B,T,R,Q,G,S<:BoxSet{R,Q,G}}
    rows = rowvals(g.mat)
    vals = nonzeros(g.mat)
    m, n = size(g)
    dict = Dict{Tuple{keytype(Q),keytype(Q)},T}()
    sizehint!(dict, length(vals))
    # iterate over columns with the keys that the columns represent
    for (col_j, j) in enumerate(g.support.set)
        for k in nzrange(g.mat, col_j)
            w = vals[k]
            row_i = rows[k]
            # grab the key that this row represents
            i = getindex_fromkeys(g, row_i)
            dict[(i,j)] = w
        end
    end
    return dict
end

# add new entries to transferoperator
function Base.checkbounds(::Type{Bool}, g::TransferOperator{B,T,S}, keys) where {B,T,S}
    all(x -> checkbounds(Bool, g.support.partition, x), keys) || return false

    diff = setdiff(keys, g.support.set)
    if !isempty(diff)
        @info(
            """
            Support of the BoxFun lies outside the already calculated
            support of the TransferOperator. Computing new transfers.
            """,
            new_keys = diff
        )

        dict = Dict(g)

        # update the support and variant set
        m, n = size(g)
        now_invariant = intersect!(copy(g.variant_set.set), diff) # ensures that order of insertion is maintained
        union!(g.support.set, now_invariant)
        setdiff!(g.variant_set.set, now_invariant)
        union!(g.support.set, diff)

        # calculate transitions for the new keys
        new_dict, new_variant_set = construct_transfers(g, BoxSet(g.support.partition, OrderedSet(diff)))
        union!(g.variant_set, new_variant_set)

        # construct the new matrix
        dict = dict ⊔ new_dict
        mat = sparse(dict, g.support, g.variant_set)
        g.mat = mat
    end
    return true
end

function Base.checkbounds(
        ::Type{Bool}, g::R, keys
    ) where {R<:Union{<:LinearAlgebra.Transpose{<:Any,<:TransferOperator},<:LinearAlgebra.Adjoint{<:Any,<:TransferOperator}}}

    all(x -> checkbounds(Bool, g.parent.support.partition, x), keys) || return false
    diff = setdiff(keys, g.parent.support.set ∪ g.parent.variant_set.set)
    if !isempty(diff)
        @warn(
            """
            support of the BoxFun lies outside of the calculated support of 
            the TransferOperator. Because the multiplication involves the 
            adjoint or transpose of the TransferOperator, lazy evaluation 
            is not possible. Consider (if possible) using the TransferOperator
            of the inverse map. 
            """, 
            adjoint_TransferOperator=g,
            invalid_keys=diff,
            maxlog=10
        )
        #return false
    end
    return true
end

function Base.checkbounds(b::Type{Bool}, g::TransferOperator, key1, key2, keys...)
    checkbounds(b, g, tuple(key1, key2, keys...))
end
