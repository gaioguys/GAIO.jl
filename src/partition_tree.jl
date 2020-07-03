struct Node
    left::Int
    right::Int
end

struct TreePartition{N,T} <: BoxPartition{Box{N,T}}
    domain::Box{N,T}
    nodes::Vector{Node}
    regular_partitions::Vector{RegularPartition{N,T}}
end

Base.copy(partition::TreePartition) = TreePartition(partition.domain, copy(partition.nodes), copy(partition.regular_partitions))

depth(partition::TreePartition) = length(partition.regular_partitions) - 1
keytype(::Type{<:TreePartition}) = Tuple{Int,Int}

function keys_all(partition::TreePartition)
    if depth(partition) != 0
        error("not implemented")
    end

    return [(0, 1)]
end

TreePartition(domain::Box) = TreePartition(domain, [Node(0, 0)], [RegularPartition(domain, 0)])

dimension(partition::TreePartition{N,T}) where {N,T} = N

function key_to_box(partition::TreePartition, key::Tuple{Int,Int})
    return key_to_box(partition.regular_partitions[key[1] + 1], key[2])
end

function unsafe_point_to_ints(partition::RegularPartition, point)
    x = (point .- partition.left) .* partition.scale
    return map(xi -> Base.unsafe_trunc(Int, xi), x)
end

function tree_search(tree::TreePartition{N,T}, point) where {N,T}
    regular_partitions = tree.regular_partitions

    # start at root
    node_idx = 1
    tree_depth = depth(tree)
    current_depth = 0
    ints = zeros(SVector{N,Int})

    while current_depth+1 <= tree_depth
        ints_next = unsafe_point_to_ints(regular_partitions[current_depth+2], point)

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
    if point_to_key(partition.regular_partitions[1], point) === nothing
        return nothing
    end

    return tree_search(partition, point)[1]
end

function subdivide!(tree::TreePartition, key::Tuple{Int,Int})
    search_key, node_idx = tree_search(tree, key_to_box(tree, key).center)

    @assert key == search_key

    node = tree.nodes[node_idx]

    @assert node.left == 0 && node.right == 0

    new_node = Node(length(tree.nodes) + 1, length(tree.nodes) + 2)
    push!(tree.nodes, Node(0, 0))
    push!(tree.nodes, Node(0, 0))

    tree.nodes[node_idx] = new_node

    if key[1] == depth(tree)
        push!(tree.regular_partitions, RegularPartition(tree.domain, depth(tree)+1))
    end

    return tree
end
