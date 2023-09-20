"""
    index_pair(F::BoxMap, N::BoxSet) -> (P₁, P₀)

Compute an index pair of `BoxSet`s P₀ ⊆ P₁ ⊆ M where M = N ∪ nbhd(N). 
"""
function index_pair(F::BoxMap, N::BoxSet)
    N = N ∪ nbhd(N)

    #F♯ = TransferOperator(F, N, N)
    #v⁺ = Vector{Float64}(undef, length(N))
    #v⁻ = Vector{Float64}(undef, length(N))

    S⁺ = maximal_forward_invariant_set(F, N; subdivision=false)#F♯, v⁺, v⁻; subdivision=false)
    S⁻ = maximal_backward_invariant_set(F, N; subdivision=false)#F♯, v⁺, v⁻: subdivision=false)

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

function isolating_neighborhood(F::BoxMap, B::BoxSet)
    N = copy(B)
    inv_N = copy(B)

    #v⁺ = Vector{Float64}(undef, length(N))
    #v⁰ = Vector{Float64}(undef, length(N))
    #v⁻ = Vector{Float64}(undef, length(N))

    while inv_N ∪ nbhd(inv_N) ⊈ N
        @debug "iteration" N=N invariant_part=inv_N
        N = union!(N, nbhd(N))
        inv_N = maximal_invariant_set(F, N; subdivision=false)#, v⁺, v⁰, v⁻; subdivision=false)
    end

    return N
end

const isolating_nbhd = isolating_neighborhood

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
    all(x -> Meta.isexpr(x, :(=)), kwargs) || throw(MethodError(var"@save", (object, kwargs...)))
    variable_name = String(object)
    return quote
        _save(
            $(esc(object)),
            generate_filename(
                $(esc(object)),
                $variable_name;
                (Pair(kwarg.args...) for kwarg in $kwargs)...
            )
        )
    end
end

suffix(::Type{T}) where {T<:TransferOperator} = ".map"
suffix(::Type{T}) where {T<:BoxSet} = ".cub"

function generate_filename(::T, variable_name; filename="", prefix="./", suffix=suffix(T)) where {T}
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

