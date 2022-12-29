# TODO: 
#       * TransferOperator compatibility
#       * cuda compatibility
#       * better cover_boxes for TreePartition
#       * avoid O(log n) time for key_to_box if ran with @inbounds ?

"""
Node structure used for `TreePartition`s

Fields:
* `left` and `right` refer to indices w.r.t. 
`trp.nodes` for a `TreePartition` `trp`. 

"""
struct Node{I}
    left::I
    right::I
end

Base.iterate(node::Node, i...) = (node.left, Val(:right))
Base.iterate(node::Node, ::Val{:right}) = (node.right, Val(:done))
Base.iterate(node::Node, ::Val{:done}) = nothing

"""
    TreePartition(domain::Box)

Binary tree structure to partition `domain` into (variably sized) boxes. 

Fields:
* `domain`: `Box` denoting the full domain.
* `nodes`:  vector of `Node`s. Each node holds two indices pointing to 
            other nodes in the vector, or 0 if the node is a leaf. 

Methods implemented:

    copy, keytype, keys, subdivide #, etc...

.
"""
struct TreePartition{N,T,I,V<:AbstractArray{Node{I}}} <: AbstractBoxPartition{Box{N,T}}
    domain::Box{N,T}
    nodes::V
end

TreePartition(domain::Box) = TreePartition(domain, [Node(0, 0)])

function TreePartition(domain::Box, depth::Integer)
    tree = TreePartition(domain)
    for i in 1:depth
        subdivide!(tree, i)
    end
    return tree
end

function BoxPartition(tree::TreePartition{N,T,I}, depth::Integer) where {N,T,I}
    center, radius = tree.domain
    dims = 2 .^ ( ((depth + N) .- (2:N+1)) .÷ N )
    BoxPartition{N,T,I}(
        tree.domain, 
        center .- radius, 
        dims ./ (2 .* radius), 
        dims
    )
end

function Base.show(io::IO, partition::TreePartition) 
    print(io, "TreePartition of depth $(depth(partition))")
end

isleaf(node::Node) = iszero(node.left) || iszero(node.right)
children(tr::TreePartition, node::Node) = isleaf(node) ? () : (tr.nodes[node.left], tr.nodes[node.right])
Base.ndims(::TreePartition{N}) where {N} = N
Base.keytype(::Type{<:TreePartition{N,T,I}}) where {N,T,I} = Tuple{I,NTuple{N,I}}
Base.copy(tr::TreePartition) = TreePartition(tr.domain, copy(tr.nodes))
Base.length(tr::TreePartition) = length(keys(tr))

function tree_search(tree::TreePartition{N,T,I}, point, max_depth=Inf) where {N,T,I}
    point in tree.domain || return nothing, 1
    
    # start at root
    P = BoxPartition(tree.domain)
    node_idx = 1
    node = tree.nodes[node_idx]
    current_depth = 1
    key = ntuple(_->one(I), Val(N))

    while !isleaf(node) && current_depth < max_depth

        # finds the CartesianIndex of the point in the "next" partition down the tree
        dim = (current_depth - 1) % N + 1
        P = subdivide(P, dim)
        key = point_to_key(P, point)
        
        # cycles through components, decides whether point lies in an even or odd box in that component
        node_idx = isodd(key[dim]) ? node.left : node.right
        node = tree.nodes[node_idx]

        current_depth += 1
    end

    key = (current_depth, key)

    return key, node_idx
end

function point_to_key(tree::TreePartition, point)
    key, _ = tree_search(tree, point)
    return key
end

@inline function key_to_box(tree::TreePartition{N}, key::Tuple{I,NTuple{N,J}}) where {N,I,J}
    depth, cart = key
    P = BoxPartition(tree, depth)
    box = key_to_box(P, cart)
    #= @boundscheck =# begin
        search_key, _ = tree_search(tree, box.center, depth)
        search_depth, search_cart = search_key
        ( search_depth < depth || search_cart != cart ) && throw(BoundsError(tree, key))
    end
    return box
end

function key_to_box(tree::TreePartition{N}, key::Tuple{I,CartesianIndex}) where {N,I}
    depth, cart = key
    key_to_box(tree, (depth, cart.I))
end

key_to_box(tree::TreePartition, ::Nothing) = nothing

"""
    subdivide!(tree::TreePartition, key::keytype(tree)) -> TreePartition
    subdivide!(tree::TreePartition, depth::Integer) -> TreePartition

    subdivide!(boxset::BoxSet{<:Any,<:Any,<:TreePartition}, key) -> BoxSet
    subdivide!(boxset::BoxSet{<:Any,<:Any,<:TreePartition}, depth) -> BoxSet

Subdivide a `TreePartition` at `key`. Dimension along which 
the node is subdivided depends on the depth of the node. 
"""
function subdivide!(tree::TreePartition{N,T,I}, key::Tuple{J,NTuple{N,K}}) where {N,T,I,J,K}
    c, _ = key_to_box(tree, key)
    _, node_idx = tree_search(tree, c)

    n = length(tree.nodes)
    tree.nodes[node_idx] = Node{I}(n+1, n+2)
    push!(tree.nodes, Node{I}(0,0), Node{I}(0,0))

    return tree
end

function subdivide!(tree::TreePartition{N,T,I}, depth::Integer) where {N,T,I}
    node_idxs = find_at_depth(tree, depth)
    leaf_idxs = union!((leaves(tree, idx) for idx in node_idxs)...)
    n = length(tree.nodes)
    for idx in leaf_idxs
        tree.nodes[idx] = Node{I}(n+1, n+2)
        push!(tree.nodes, Node{I}(0,0), Node{I}(0,0))
        n = n + 2
    end
    return tree
end

function subdivide(tree::TreePartition, key_or_depth)
    subdivide!(copy(tree), key_or_depth)
end

"""
Return the depth of the tree structure. 
"""
function depth(tree::TreePartition{N,T,I}) where {N,T,I}
    depth = 1
    queue = Tuple{Int,I}[(1, 1)]
    while !isempty(queue)
        current_depth, node_idx = pop!(queue)
        node = tree.nodes[node_idx]
        depth = max(depth, current_depth)
        if !isleaf(node)
            c1_idx, c2_idx = node
            push!(queue, (current_depth+1, c1_idx), (current_depth+1, c2_idx))
        end
    end
    return depth
end

function Base.size(tree::TreePartition{N,T,I}) where {N,T,I}
    depth = 1
    sizes = Dict{Int,Int}()
    queue = Tuple{Int,I}[(1, 1)]
    while !isempty(queue)
        current_depth, node_idx = pop!(queue)
        node = tree.nodes[node_idx]
        sizes[current_depth] = get(sizes, current_depth, 0) + 1
        depth = max(depth, current_depth)
        if !isleaf(node)
            c1_idx, c2_idx = node
            push!(queue, (current_depth+1, c1_idx), (current_depth+1, c2_idx))
        end
    end
    return ntuple(i->get(sizes, i, 1), depth)
end

function find_at_depth(tree::TreePartition{N,T,I}, depth::Integer) where {N,T,I}
    node_idxs = I[]
    queue = Tuple{Int,I}[(1, 1)]
    while !isempty(queue)
        current_depth, node_idx = pop!(queue)
        if current_depth == depth
            push!(node_idxs, node_idx)
        else
            c1_idx, c2_idx = tree.nodes[node_idx]
            push!(queue, (current_depth+1, c1_idx), (current_depth+1, c2_idx))
        end
    end
   return node_idxs
end

function leaves(tree::TreePartition{N,T,I}, initial_node_idx=1) where {N,T,I}
    leaf_idxs = I[]
    queue = I[initial_node_idx]
    while !isempty(queue)
        node_idx = pop!(queue)
        node = tree.nodes[node_idx]
        if isleaf(node)
            push!(leaf_idxs, node_idx)
        else
            c1_idx, c2_idx = node
            push!(queue, c1_idx, c2_idx)
        end
    end
    return leaf_idxs
end

function Base.keys(tree::Q) where {N,T,I,Q<:TreePartition{N,T,I}}
    K = keytype(Q)
    keys = K[]
    queue = Tuple{I,K}[(1, K((1, ntuple(_->1,N))))]
    while !isempty(queue)
        node_idx, key = pop!(queue)
        node = tree.nodes[node_idx]
        if isleaf(node)
            push!(keys, key)
        else
            c1_idx, c2_idx = node
            depth, cart = key
            dim = (depth - 1) % N + 1
            key1 = (depth+1, Base.setindex(cart, 2 * cart[dim] - 1, dim))
            key2 = (depth+1, Base.setindex(cart, 2 * cart[dim], dim))
            push!(queue, (c1_idx, key1), (c2_idx, key2))
        end
    end
    return keys
end

function Base.:(==)(tr1::TreePartition{N,T,I}, tr2::TreePartition{N,V,J}) where {N,T,I,V,J}
    tr1.domain == tr2.domain || return false
    queue = Tuple{I,J}[(1, 1)]
    while !isempty(queue)
        idx1, idx2 = pop!(queue)
        node1, node2 = tr1.nodes[idx1], tr2.nodes[idx2]
        if !isleaf(node1) && !isleaf(node2)
            c1_1, c2_1 = node1
            c1_2, c2_2 = node2
            push!(queue, (c1_1, c1_2), (c2_1, c2_2))
        elseif !isleaf(node1) || !isleaf(node2)
            return false
        end
    end
    return true
end
