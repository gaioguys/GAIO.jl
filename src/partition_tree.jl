# each node holds the indices of the two partitions that it sends to
# indices wrt to TreePartition.regular_partitions
struct Node
    left::Int
    right::Int
end

struct TreePartition{N,T} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    nodes::Vector{Node}
    regular_partitions::Vector{BoxPartition{N,T}}
end

Base.copy(partition::TreePartition) = TreePartition(partition.domain, copy(partition.nodes), copy(partition.regular_partitions))

depth(partition::TreePartition) = length(partition.regular_partitions) - 1
Base.keytype(::Type{<:TreePartition}) = Tuple{Int,Int}

function Base.show(io::IO, partition::TreePartition) 
    print(io, "TreePartition of depth $(depth(partition))")
end

function Base.:(==)(p1::TreePartition, p2::TreePartition)
    @debug "equality between TreePartitions is not implemented" maxlog=1
    return true
end

function Base.keys(partition::TreePartition)
    if depth(partition) != 0
        error("not implemented")
    end

    return [(0, 1)]
end

TreePartition(domain::Box) = TreePartition(domain, [Node(0, 0)], [BoxPartition(domain)])

Base.ndims(::TreePartition{N,T}) where {N,T} = N

# TreePartition keys are of the form (partition_key, point_key_in_said_partition)
function key_to_box(partition::TreePartition, key::Tuple{Int,Int})
    return key_to_box(partition.regular_partitions[key[1] + 1], key[2])
end

function tree_search(tree::TreePartition{N,T}, point) where {N,T}
    regular_partitions = tree.regular_partitions

    # start at root
    node_idx = 1
    tree_depth = depth(tree)
    current_depth = 0
    ints = zeros(SVector{N,Int})

    while current_depth < tree_depth
        # finds the CartesianIndex of the point in the "next" partition down the tree
        ints_next = unsafe_point_to_ints(regular_partitions[current_depth+2], point)

        # cycles through components, decides whether point lies in an even or odd box in that component
        point_is_left = iseven(ints_next[(current_depth % N) + 1])

        
        node = tree.nodes[node_idx]
        node_idx_next = ifelse(point_is_left, node.left, node.right)

        if node_idx_next == 0
            break
        end

        node_idx = node_idx_next

        ints = ints_next
        current_depth += 1
    end

    key = (current_depth, sum(ints .* regular_partitions[current_depth+1].dimsprod) + 1)

    return key, node_idx
end

function point_to_key(partition::TreePartition, point)
    if isnothing(point_to_key(partition.regular_partitions[1], point))
        return nothing
    end

    return tree_search(partition, point)[1]
end

function subdivide!(tree::TreePartition{N, T}, key::Tuple{Int,Int}) where {N, T}
    search_key, node_idx = tree_search(tree, key_to_box(tree, key).center)

    key != search_key && throw(BoundsError(tree, key))

    node = tree.nodes[node_idx]

    if node.left != 0 || node.right != 0
        error("Subdivide along non-leaf nodes is not implemented")
    end

    new_node = Node(length(tree.nodes) + 1, length(tree.nodes) + 2)
    push!(tree.nodes, Node(0, 0))
    push!(tree.nodes, Node(0, 0))

    tree.nodes[node_idx] = new_node

    if key[1] == depth(tree)
        push!(tree.regular_partitions, BoxPartition(tree.domain, depth=depth(tree)+1 #=ntuple(i -> depth(tree)+1, N)=#))
    end

    return tree
end
