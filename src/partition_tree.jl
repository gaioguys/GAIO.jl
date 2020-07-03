struct Node
    left::Int
    right::Int
end

mutable struct TreePartition{N,T} <: BoxPartition{Box{N,T}}
    domain::Box{N,T}
    nodes::Vector{Node}
    depth::Int
end

Base.copy(partition::TreePartition) = TreePartition(partition.domain, copy(partition.nodes), partition.depth)

keytype(::Type{<:TreePartition}) = Tuple{Int,Int}

function keys_all(partition::TreePartition)
    if partition.depth != 0
        error("not implemented")
    end

    return [(0, 1)]
end

TreePartition(domain::Box) = TreePartition(domain, [Node(0, 0)], 0)

dimension(partition::TreePartition{N,T}) where {N,T} = N

function Base.getindex(partition::TreePartition, index::Tuple{Int,Int})
    return RegularPartition(partition.domain, index[1])[index[2]]
end

struct TreePartitionKeys{N,T}
    tree::TreePartition{N,T}
    regular_keys::Vector{RegularPartitionKeys{N,T}}
end

function Base.keys(partition::TreePartition{N,T}) where {N,T}
    return TreePartitionKeys(
        partition,
        map(depth -> keys(RegularPartition(partition.domain, depth)), 0:partition.depth)
    )
end

function unsafe_point_to_ints(keys::RegularPartitionKeys, point)
    x = (point .- keys.left) .* keys.scale
    return map(xi -> Base.unsafe_trunc(Int, xi), x)
end

function tree_search(keys::TreePartitionKeys{N,T}, point) where {N,T}
    regular_keys = keys.regular_keys
    tree = keys.tree

    # start at root
    node_idx = 1
    depth = 0
    ints = zeros(SVector{N,Int})

    while depth+1 <= tree.depth
        ints_next = unsafe_point_to_ints(regular_keys[depth+2], point)

        point_is_left = iseven(ints_next[(depth % N) + 1])

        node = tree.nodes[node_idx]
        node_idx_next = ifelse(point_is_left, node.left, node.right)

        if node_idx_next == 0
            break
        end

        node_idx = node_idx_next

        ints = ints_next
        depth += 1
    end

    key = (depth, sum(ints .* regular_keys[depth+1].dimsprod) + 1)

    return key, node_idx
end

function Base.getindex(keys::TreePartitionKeys, point)
    if keys.regular_keys[1][point] === nothing
        return nothing
    end

    return tree_search(keys, point)[1]
end

function subdivide!(tree::TreePartition, key::Tuple{Int,Int})
    search_key, node_idx = tree_search(keys(tree), tree[key].center)

    @assert key == search_key

    node = tree.nodes[node_idx]

    @assert node.left == 0 && node.right == 0

    new_node = Node(length(tree.nodes) + 1, length(tree.nodes) + 2)
    push!(tree.nodes, Node(0, 0))
    push!(tree.nodes, Node(0, 0))

    tree.nodes[node_idx] = new_node

    if key[1] == tree.depth
        tree.depth += 1
    end

    return tree
end
