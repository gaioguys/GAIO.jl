struct BoxList{P<:AbstractBoxPartition,L<:AbstractVector}
    partition::P
    keylist::L
end

Base.length(list::BoxList) = length(list.keylist)
Base.getindex(list::BoxList, x::AbstractVector) = BoxList(list.partition, list.keylist[x])
Base.getindex(list::BoxList, i::Int64) = list.keylist[i]

BoxList(set::BoxSet) = BoxList(set.partition, collect(set.set))

BoxSet(list::BoxList) = BoxSet(list.partition, Set(list.keylist))

"""
    TransferOperator(map::BoxMap, support::BoxSet, mat::SparseMatrixCSC)
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

It is important to note that `TranferOperator` is only supported over the 
box set `B`, but if one lets a `TranferOperator` act on a `BoxFun`, then 
the support `B` is extended "on the fly" to include the support of the `BoxFun`.

Methods Implemented: 
```julia
:(==), axes, size, eltype, getindex, setindex!, SparseArrays.sparse, Arpack.eigs, LinearAlgebra.mul! #, etc ...
```
.
"""
struct TransferOperator{L<:BoxList,W}
    vertices::L
    edges::Dict{Tuple{Int,Int},W}
end

function Base.show(io::IO, T::TransferOperator)
    n = length(T.vertices)
    m = length(T.edges)
    print(io, "TransferOperator on $(n) boxes with $(m) edges")
end

function invert_vector(x::AbstractVector{T}) where T
    dict = Dict{T,Int}()
    sizehint!(dict, length(x))

    for i in 1:length(x)
        dict[x[i]] = i
    end

    return dict
end

# TODO: this code is generally incorrect. only valid for BoxPartition and special choices of points
function TransferOperator(g::SampledBoxMap, boxset::BoxSet)
    Q = g.domain    
    n = length(g.domain_points(Q.center, Q.radius))
    P = boxset.partition
    edges = [ Dict{Tuple{Int64,Int64},Float64}() for k = 1:nthreads() ]
    boxlist = BoxList(boxset)
    key_to_index = invert_vector(boxlist.keylist)

    @threads for i = 1:length(boxlist)
        box = key_to_box(P, boxlist[i])
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = g.map(@muladd p .* r .+ c)
            hit = point_to_key(P, fp)
            if !isnothing(hit)
                if hit in boxset.set
                    j = key_to_index[hit]
                    e = (i,j)
                    edges[threadid()][e] = get(edges[threadid()], e, 0) + 1.0/n
                end
            end
        end
    end
    edges = merge(edges...)

    return TransferOperator(boxlist, edges)
end


function Graphs.strongly_connected_components(gstar::TransferOperator)
    graph = SimpleDiGraph(
        [Edge(edge[1], edge[2]) for edge in keys(gstar.edges)]
    )

    sccs = Graphs.strongly_connected_components(graph)

    connected_vertices = Int[]

    for scc in sccs
        if length(scc) > 1 || has_edge(graph, scc[1], scc[1])
            append!(connected_vertices, scc)
        end
    end

    return BoxSet(gstar.vertices[connected_vertices])
end

function matrix(gstar::TransferOperator{L,W}) where {L,W}
    num_edges = length(gstar.edges)

    I = Vector{Int}()
    sizehint!(I, num_edges)

    J = Vector{Int}()
    sizehint!(J, num_edges)

    V = Vector{W}()
    sizehint!(V, num_edges)

    for (edge, weight) in gstar.edges
        push!(I, edge[1])
        push!(J, edge[2])
        push!(V, weight)
    end

    n = length(gstar.vertices)
    return sparse(I, J, V, n, n)
end


"""
    eigs(gstar::TransferOperator [; kwargs...]) -> (d[, v], nconv)

Compute a set of eigenvalues `d` and eigenmeasures `v` of `gstar`. 
Works with the adjoint _Koopman operator_ as well. 
All keyword arguments from `Arpack.eigs` can be passed. See the 
documentation for `Arpack.eigs`. 
"""
function Arpack.eigs(gstar::TransferOperator{BoxList{P,L}}; nev::Int=1) where {P,L}
    G = matrix(gstar)

    λ, ϕ, nconv = eigs(G'; nev=nev)

    ϕ_funs = BoxFun{P,keytype(P),ComplexF64}[]

    for i in 1:nev
        ϕi = ϕ[:,i]
        dict = Dict{keytype(P),ComplexF64}()
        sizehint!(dict, length(ϕi))

        for j in 1:length(ϕi)
            dict[gstar.vertices.keylist[j]] = ϕi[j]
        end

        ϕ_fun = BoxFun{P,keytype(P),ComplexF64}(gstar.vertices.partition, dict)
        normalize!(ϕ_fun)
        push!(ϕ_funs, ϕ_fun)
    end

    return λ, ϕ_funs, nconv
end
