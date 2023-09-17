"""
helper function to compute intersections
"""
⊓(a, b) = a != 0 && b != 0

"""
    fast_forward_invariant_part(F♯::TransferOperator, v⁺, v⁻) -> BoxSet

Given a TransferOperator and preallocated vectors `v⁺`, `v⁻`
(representing boolean vectors as subsets of `F♯.domain`), 
compute the subset of `F♯.domain` which is forward invariant. 

This is equivalent to calling 
```julia
maximal_forward_invariant_set(F, F♯.domain)
```

.
"""
function fast_forward_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁺, 1); fill(v⁻, 0)
    M = F♯.mat
    while v⁺ != v⁻
        v⁻ .= v⁺
        v⁺ .= M*v⁻
        v⁺ .= v⁺ .⊓ v⁻
    end
    return BoxSet(F♯.domain, v⁺ .!= 0)
end

"""
    fast_backward_invariant_part(F♯::TransferOperator, v⁺, v⁻) -> BoxSet

Given a TransferOperator and preallocated vectors `v⁺`, `v⁻`
(representing boolean vectors as subsets of `F♯.domain`), 
compute the subset of `F♯.domain` which is backward invariant. 

This is equivalent to calling 
```julia
maximal_backward_invariant_set(F, F♯.domain)
```

.
"""
function fast_backward_invariant_part(F♯, v⁺, v⁻)
    fill!(v⁻, 1); fill!(v⁺, 0)
    M = F♯.mat
    while v⁻ != v⁺
        v⁺ .= v⁻
        v⁻ .* M'v⁺
        v⁻ .= v⁺ .⊓ v⁻
    end
    return BoxSet(F♯.domain, v⁻ .!= 0)
end

"""
    index_pair(F::BoxMap, N::BoxSet) -> (P₁, P₀)

Compute an index pair of `BoxSet`s P₀ ⊆ P₁ ⊆ M where M = N ∪ nbhd(N). 
"""
function index_pair(F::BoxMap, N::BoxSet)
    N = N ∪ nbhd(N)

    F♯ = TransferOperator(F, N, N)
    v⁺ = Vector{Float64}(undef, length(N))
    v⁻ = Vector{Float64}(undef, length(N))

    S⁺ = fast_forward_invariant_part(F♯, v⁺, v⁻)
    S⁻ = fast_backward_invariant_part(F♯, v⁺, v⁻)

    P₁ = S⁻
    P₀ = setdiff(S⁻, S⁺)
    return P₁, P₀
end

"""
    index_quad(F::BoxMap, N::BoxSet) -> (P₁, P₀, P̄₁, P̄₀)

Compute a tuple of index pairs such that 
`F: (P₁, P₀) → (P̄₁, P̄₀)`
"""
function index_quad(F::BoxMap, N::BoxSet)
    P₁, P₀ = index_pair(F, N)
    FP₁ = F(P₁)
    P̄₁ = P₁ ∪ FP₁
    P̄₀ = P₀ ∪ setdiff(FP₁, P₁)
    return P₁, P₀, P̄₁, P̄₀
end

"""
    @save boxset prefix="./" suffix=".boxset" -> filename
    @save boxset filename -> filename

Save a `BoxSet` as a list of keys. The default file name is the 
variable name. 

Note that this does not include information on the 
partition of the `BoxSet`, just the keys. 

.

    @save boxmap source prefix="./" suffix=".boxmap" -> filename
    @save boxmap source filename -> filename

Save a `BoxMap` as a list of source-keys and their image-keys in the form
```
key_1 -> {image_1, image_2, image_3}
key_2 -> {image_2, image_4, image_8, image_6}
⋮
```

.

    @save transfer_operator prefix="./" suffix=".boxmap" -> filename
    @save transfer_operator filename -> filename

Save a `TransferOperator` as a list of keys and their image-keys in the form
```
key_1 -> {image_1, image_2, image_3}
key_2 -> {image_2, image_4, image_8, image_6}
⋮
```
"""
macro save(object, kwargs...)
    all(x -> Meta.isexpr(x, :(=)), kwargs) || throw(MethodError(var"@save", (F♯, kwargs...)))
    variable_name = String(object)
    return quote
        _save(
            $(esc(object)),
            generate_filename($variable_name; (Pair(kwarg.args...) for kwarg in $kwargs)...)
        )
    end
end

function generate_filename(variable_name; filename="", prefix::String="./", suffix::String=".dat")
    isempty(filename) ? prefix * variable_name * suffix : filename
end

function _save(boxset::BoxSet, filename)
    file = open(filename, "w")
    for key in boxset.set
        println(file, key)
    end
    close(file)
    return filename
end

function _save(F♯::TransferOperator, filename)
    _rehash!(F♯)
    file = open(filename, "w")

    adj = F♯.mat
    dom = F♯.domain.set
    codom = F♯.codomain.set
    rows = rowvals(adj)

    for (col_j, key_j) in enumerate(dom)
        print(file, "$key_j -> {")
        join(file, (index_to_key(codom, rows[i]) for i in nzrange(adj, col_j)), ", ")
        println(file, "}")
    end

    close(file)
    return filename
end

function _save(F::BoxMap, source::BoxSet, filename)
    _save(TransferOperator(F, source), filename)
end
