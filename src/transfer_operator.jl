struct BoxList{P<:AbstractBoxPartition,L<:AbstractVector}
    partition::P
    keylist::L
end

Base.length(list::BoxList) = length(list.keylist)
Base.getindex(list::BoxList, x::AbstractVector) = BoxList(list.partition, list.keylist[x])
Base.getindex(list::BoxList, i::Int64) = list.keylist[i]

BoxList(set::BoxSet) = BoxList(set.partition, collect(set.set))

BoxSet(list::BoxList) = BoxSet(list.partition, Set(list.keylist))

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
            fp = g.map(c.+r.*p)
            hit = point_to_key(P, fp)
            if hit !== nothing
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


function strongly_connected_components(gstar::TransferOperator)
    graph = SimpleDiGraph(
        [Edge(edge[1], edge[2]) for edge in keys(gstar.edges)]
    )

    sccs = LightGraphs.strongly_connected_components(graph)

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
