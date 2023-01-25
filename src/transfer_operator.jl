"""
    TransferOperator(map::BoxMap, domain::BoxSet)
    TransferOperator(map::BoxMap, domain::BoxSet, codomain::BoxSet)

Discretization of the Perron-Frobenius operator, or transfer operator. 
Implemented as a sparse matrix with indices referring to 
two `BoxSet`s: `domain` and `codomain`. 

There exists two constructors:
* only provide a `boxmap` and a `domain`. In this case, 
  the `codomain` is generated as the image of `domain` under 
  the `boxmap`. 
  ```julia
  julia> P = BoxPartition( Box((0,0), (1,0)), (10,10) )
    10 x 10 - element BoxPartition
  
  julia> domain = BoxSet( P, Set((1,2), (2,3), (3,4)) )
    3 - element Boxset over 10 x 10 - element BoxPartition
  
  julia> T = TransferOperator(boxmap, domain)
    TransferOperator over [...]
  ```
* provide `domain` and `codomain`. In this case, 
  the size of the transition matrix is given. 
  ```julia
  julia> codomain = domain
    3 - element Boxset over 10 x 10 - element BoxPartition
  
  julia> T = TransferOperator(boxmap, domain, codomain)
    TransferOperator over [...]
  ```

Fields:
* `mat`:            `SparseMatrixCSC` containing transfer weights. The index 
                    `T.mat[i,j]` represents the transfer weight FROM the `j`'th
                    box in `codomain` TO the `i`'th box in `domain`. 
* `boxmap`:         `SampledBoxMap` map which dictates the transfer weights. 
* `domain`:         `BoxSet` which contains keys for the already calculated transfers. 
                    Effectively, these are column pointers, i.e. the 
                    `j`th column of `T.mat` contains transfer weights FROM 
                    box B_j, where B_j is the `j`th box of `domain`. 
* `codomain`:       `BoxSet` which contains keys for the already calculated transfers. 
                    Effectively, these are row pointers, i.e. the 
                    `i`th row of `T.mat` contains transfer weights TO 
                    box B_i, where B_i is the `i`th box of `codomain`. 

```julia
        domain -->
codomain  .   .   .   .   .
    |     .   .   .   .   .
    |     .   .   .   .   .
    v     .   .  mat  .   .
          .   .   .   .   .
          .   .   .   .   .
          .   .   .   .   .
          .   .   .   .   .
```

It is important to note that `TranferOperator` is only supported over the 
box set `domain`, but if one lets a `TranferOperator` act on a `BoxFun`, e.g. 
by multiplication, then the `domain` is extended "on the fly" to 
include the support of the `BoxFun`.

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
function TransferOperator(g::BoxMap, domain::BoxSet{B,P,S}) where {B,P,S}
    dom = BoxSet(domain.partition, OrderedSet(domain.set))
    TransferOperator(g, dom)
end

function TransferOperator(g::BoxMap, domain::BoxSet{B,P,S}, codomain::BoxSet{R,Q,W}) where {B,P,S,R,Q,W}
    dom = BoxSet(domain.partition, OrderedSet(domain.set))
    codom = BoxSet(codomain.partition, OrderedSet(codomain.set))
    TransferOperator(g, dom, codom)
end

function TransferOperator(
        g::BoxMap, domain::BoxSet{B,P,S}
    ) where {B,P,S<:OrderedSet}

    dict, codomain = construct_transfers(g, domain)
    mat = sparse(dict, domain, codomain)

    for i in 1:size(mat, 2)
        s = sum(mat[:, i])
        iszero(s) || mat[:, i] ./= s
    end

    return TransferOperator(g, domain, codomain, mat)
end

function TransferOperator(
        g::BoxMap, domain::BoxSet{B,P,S}, codomain::BoxSet{R,Q,W}
    ) where {B,P,S<:OrderedSet,R,Q,W<:OrderedSet}

    dict = construct_transfers(g, domain, codomain)
    mat = sparse(dict, domain, codomain)

    for i in 1:size(mat, 2)
        s = sum(mat[:, i])
        iszero(s) || (mat[:, i] ./= s)
    end

    return TransferOperator(g, domain, codomain, mat)
end

function Base.show(io::IO, g::TransferOperator)
    print(io, "TransferOperator over $(g.domain)")
end

function Base.show(io::IO, ::MIME"text/plain", g::TransferOperator)
    print(io, "TransferOperator over $(g.domain) with $(length(nonzeros(g.mat))) stored entries:\n\n")
    SparseArrays._show_with_braille_patterns(io, g.mat)
end

Base.:(==)(g1::TransferOperator, g2::TransferOperator) = g1.mat == g2.mat
Base.eltype(::Type{<:TransferOperator{B,T}}) where {B,T} = T
Base.keytype(::Type{<:TransferOperator{B,T,I}}) where {B,T,I} = I
Base.size(g::TransferOperator) = size(g.mat)
Base.Matrix(g::TransferOperator) = Matrix(g.mat)
SparseArrays.sparse(g::TransferOperator) = copy(g.mat)
Base.axes(g::TransferOperator) = (collect(g.domain), collect(g.codomain))

@propagate_inbounds function Base.getindex(g::TransferOperator{T,I}, u, v) where {T,I}
    @boundscheck checkbounds(Bool, g, u, v) || throw(BoundsError(g, (u,v)))
    i, j = getkeyindex(g.codomain, u), getkeyindex(g.domain, v)
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

        function _eigs(g::$type, B=I; nev=1, ritzvec=true, kwargs...)
            λ, ϕ, nconv = Arpack._eigs(g, B; nev=nev, ritzvec=true, kwargs...)
            S = $gmap.domain
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
        Arpack.eigs(g::$type, B::UniformScaling=I; kwargs...) = _eigs(g, B; kwargs...)
        Arpack.eigs(g::$type, B; kwargs...) = _eigs(g, B; kwargs...)

        LinearAlgebra.mul!(y::AbstractVector, g::$type, x::AbstractVector) = mul!(y, $func($gmap.mat), x)

        @propagate_inbounds function LinearAlgebra.mul!(y::BoxFun, g::$type, x::BoxFun)
            @boundscheck(checkbounds(Bool, g, keys(x.vals)) || throw(BoundsError(g, x)))
            zer = zero(eltype(y))
            map!(x -> zer, values(y.vals))
            rows = rowvals($gmap.mat)
            vals = nonzeros($gmap.mat)
            # iterate over columns with the keys that the columns represent
            for (col_j, j) in enumerate($gmap.domain.set)
                for k in nzrange($gmap.mat, col_j)
                    w = vals[k]
                    row_i = rows[k]
                    # grab the key that this row represents
                    i = getindex_fromkeys($gmap.codomain, row_i)
                    y[$ind1] = @muladd y[$ind1] + $func(w) * x[$ind2]
                end
            end
            return y
        end

        function Base.:(*)(g::$type, x::BoxFun{B,K,V}) where {B,K,V}
            dict = OrderedDict{K, promote_type(eltype(g), V)}()
            sizehint!(dict, length($gmap.codomain))
            y = BoxFun($gmap.codomain.partition, dict)
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
# converts partition-key to index if a set is enumerated 1..n, or nothing if key not in set
getkeyindex(dict::Dict, i) = (j = Base.ht_keyindex(dict, i); j > 0 ? j : nothing)
getkeyindex(set::Set, i) = (j = Base.ht_keyindex(set.dict, i); j > 0 ? j : nothing)
getkeyindex(dict::OrderedDict, i) = (j = OrderedCollections.ht_keyindex(dict, i, true); j > 0 ? j : nothing)
getkeyindex(set::OrderedSet, i) = (j = OrderedCollections.ht_keyindex(set.dict, i, true); j > 0 ? j : nothing)
getkeyindex(boxset::BoxSet, i) = getkeyindex(boxset.set, i)

# converts index if a set is enumerated 1..n to partition-key, or BoundsError
getindex_fromkeys(dict::Union{Dict,OrderedDict}, i) = dict.keys[i]
getindex_fromkeys(set::Union{Set,OrderedSet}, i) = getindex_fromkeys(set.dict, i)
getindex_fromkeys(boxset::BoxSet, i) = getindex_fromkeys(boxset.set, i)
getindex_fromkeys(g::TransferOperator, i, j) = (getindex_fromkeys(g.codomain, i), getindex_fromkeys(g.domain, j))

# convert DOK sparse matrix to CSC using indices from support / variant_set
function SparseArrays.sparse(
        dict::Dict{Tuple{I,I},T}, domain::BoxSet, codomain::BoxSet
    ) where {I,T}

    _rehash!(domain)
    _rehash!(codomain)

    m, n = length(codomain), length(domain)
    xs, ys, ws = Int[], Int[], T[]

    sizehint!(xs, length(dict))
    sizehint!(ys, length(dict))
    sizehint!(ws, length(dict))

    for ((u, v), w) in dict
        x = getkeyindex(codomain, u)
        y = getkeyindex(domain, v)
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
    for (col_j, j) in enumerate(g.domain.set)
        for k in nzrange(g.mat, col_j)
            w = vals[k]
            row_i = rows[k]
            # grab the key that this row represents
            i = getindex_fromkeys(g.codomain, row_i)
            dict[(i,j)] = w
        end
    end
    return dict
end

# add new entries to transferoperator
function Base.checkbounds(::Type{Bool}, g::TransferOperator{B,T,S}, keys) where {B,T,S}
    all(x -> checkbounds(Bool, g.domain.partition, x), keys) || return false

    diff = setdiff(keys, g.domain.set)
    if !isempty(diff)
        @info(
            """
            Support of the BoxFun lies outside the already calculated
            support of the TransferOperator. Computing new transfers.
            """,
            new_keys = diff
        )

        dict = Dict(g)
        m, n = size(g)

        # calculate transitions for the new keys
        new_dict, new_codomain = construct_transfers(g, BoxSet(g.domain.partition, OrderedSet(diff)))
        union!(g.codomain, new_codomain)

        # construct the new matrix
        dict = dict ⊔ new_dict
        mat = sparse(dict, g.domain, g.codomain)
        g.mat = mat
    end
    return true
end

function Base.checkbounds(
        ::Type{Bool}, g::R, keys
    ) where {R<:Union{<:LinearAlgebra.Transpose{<:Any,<:TransferOperator},<:LinearAlgebra.Adjoint{<:Any,<:TransferOperator}}}

    all(x -> checkbounds(Bool, g.parent.codomain.partition, x), keys) || return false
    diff = setdiff(keys, g.parent.codomain.set)
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
