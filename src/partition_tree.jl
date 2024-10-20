"""
Node structure used for `BoxTree`s

Fields:
* `left` and `right` refer to indices w.r.t. 
`trp.nodes` for a `BoxTree` `trp`. 

"""
struct Node{I}
    left::I
    right::I
end

Base.iterate(node::Node, i...) = (node.left, Val(:right))
Base.iterate(node::Node, ::Val{:right}) = (node.right, Val(:done))
Base.iterate(node::Node, ::Val{:done}) = nothing

"""
    BoxTree(domain::Box)

Binary tree structure to partition `domain` into (variably sized) boxes. 

Fields:
* `domain`: `Box` denoting the full domain.
* `nodes`:  vector of `Node`s. Each node holds two indices pointing to 
            other nodes in the vector, or 0 if the node is a leaf. 

Methods implemented:

    copy, keytype, keys, subdivide #, etc...

.
"""
struct BoxTree{N,T,I,V<:AbstractArray{Node{I}}} <: BoxLayout{Box{N,T}}
    domain::Box{N,T}
    nodes::V
end

function BoxTree(domain::Box, depth::Integer)
    tree = BoxTree(domain)
    for i in 1:depth
        subdivide!(tree, i)
    end
    return tree
end

BoxTree(domain::Box) = BoxTree(domain, [Node(0, 0)])
BoxGrid(tree::BoxTree) = BoxGrid(tree, depth(tree))

function BoxGrid(tree::BoxTree{N,T,I}, depth::Integer) where {N,T,I}
    center, radius = tree.domain
    dims = ntuple(Val(N)) do i
        2 ^ ( ((depth+N) - (i+1)) ÷ N )
    end
    #dims = 2 .^ ( ((depth + N) .- (2:N+1)) .÷ N )
    BoxGrid{N,T,I}(
        tree.domain, 
        center .- radius, 
        dims ./ (2 .* radius), 
        dims
    )
end

function Base.show(io::IO, partition::BoxTree) 
    print(io, "BoxTree of depth $(depth(partition))")
end

isleaf(node::Node) = iszero(node.left) || iszero(node.right)
children(tr::BoxTree, node::Node) = isleaf(node) ? () : (tr.nodes[node.left], tr.nodes[node.right])
center(tr::BoxTree) = center(tr.domain)
radius(tr::BoxTree) = radius(tr.domain)

Base.ndims(::BoxTree{N}) where {N} = N
Base.keytype(::Type{<:BoxTree{N,T,I}}) where {N,T,I} = Tuple{I,NTuple{N,I}}
Base.copy(tr::BoxTree) = BoxTree(tr.domain, copy(tr.nodes))
Base.length(tr::BoxTree) = length(keys(tr))
Base.sizehint!(tr::BoxTree, s) = sizehint!(tr.nodes, s)

"""
    tree_search(tree, point, max_depth=Inf) -> key, node_idx

Find the key and correspoinding node index within the tree 
data structure containing a point. 
"""
function tree_search(tree::TR, point, max_depth=Inf) where {N,T,I,TR<:BoxTree{N,T,I}}
    point in tree.domain || return nothing, 1
    
    # start at root
    P = BoxGrid{I}(tree.domain)
    node_idx = 1
    node = tree.nodes[node_idx]
    current_depth = one(I)
    cart = ntuple(_->one(I), Val(N))    # = point_to_key(P, point)

    while !isleaf(node) && current_depth < max_depth

        # finds the CartesianIndex of the point in the "next" partition down the tree
        dim = (current_depth - 1) % N + 1
        P = subdivide(P, dim)
        cart = point_to_key(P, point)
        
        # cycles through components, decides whether point lies in an even or odd box in that component
        node_idx = isodd(cart[dim]) ? node.left : node.right
        node = tree.nodes[node_idx]

        current_depth += one(I)
    end

    key = (current_depth, cart) :: keytype(TR)

    return key, node_idx
end

function Base.checkbounds(::Type{Bool}, tree::BoxTree, key)
    depth, cart = key
    P = BoxGrid(tree, depth)
    c, _ = key_to_box(P, cart)
    search_key, _ = tree_search(tree, c, depth+1)
    search_depth, search_cart = search_key
    return search_depth > depth || ( search_depth == depth && search_cart == cart )
end

function point_to_key(tree::BoxTree, point)
    key, _ = tree_search(tree, point)
    return key
end

@propagate_inbounds function key_to_box(tree::BoxTree{N}, key::Tuple{I,NTuple{N,J}}) where {N,I,J}
    @boundscheck checkbounds(Bool, tree, key) || throw(BoundsError(tree, key))
    depth, cart = key
    P = BoxGrid(tree, depth)
    box = key_to_box(P, cart)
    return box
end

function key_to_box(tree::BoxTree{N}, key::Tuple{I,CartesianIndex}) where {N,I}
    depth, cart = key
    key_to_box(tree, (depth, cart.I))
end

key_to_box(tree::BoxTree, ::Nothing) = nothing

"""
    subdivide!(tree::BoxTree, key::keytype(tree)) -> BoxTree
    subdivide!(tree::BoxTree, depth::Integer) -> BoxTree

    subdivide!(boxset::BoxSet{<:Any,<:Any,<:BoxTree}, key) -> BoxSet
    subdivide!(boxset::BoxSet{<:Any,<:Any,<:BoxTree}, depth) -> BoxSet

Subdivide a `BoxTree` at `key`. Dimension along which 
the node is subdivided depends on the depth of the node. 
"""
@propagate_inbounds function subdivide!(tree::BoxTree{N,T,I}, key::Tuple{J,NTuple{N,K}}) where {N,T,I,J,K}
    depth, cart = key
    P = BoxGrid(tree, depth)
    c, _ = key_to_box(P, cart)
    search_key, node_idx = tree_search(tree, c, depth + 1)
    
    @boundscheck begin
        search_depth, search_cart = search_key
        search_depth > depth || ( search_depth == depth && search_cart == cart ) || throw(BoundsError(tree, key))
    end

    n = length(tree.nodes)
    tree.nodes[node_idx] = Node{I}(n+1, n+2)
    push!(tree.nodes, Node{I}(0,0), Node{I}(0,0))

    return tree
end

function subdivide!(tree::BoxTree{N,T,I}, depth::Integer=1) where {N,T,I}
    node_idxs = find_at_depth(tree, depth)

    if all(idx -> isleaf(tree.nodes[idx]), node_idxs)
        leaf_idxs = node_idxs
    else
        leaf_idxs = union!((leaves(tree, idx) for idx in node_idxs)...)
    end

    n = length(tree.nodes)
    sizehint!(tree, n + 2*length(leaf_idxs))

    for idx in leaf_idxs
        tree.nodes[idx] = Node{I}(n+1, n+2)
        push!(tree.nodes, Node{I}(0,0), Node{I}(0,0))
        n = n + 2
    end

    return tree
end

function subdivide(tree::BoxTree, key_or_depth)
    subdivide!(copy(tree), key_or_depth)
end

"""
    depth(tree::BoxTree)

Return the depth of the tree structure. 
"""
function depth(tree::BoxTree{N,T,I}) where {N,T,I}
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

function Base.size(tree::BoxTree{N,T,I}) where {N,T,I}
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

"""
    find_at_depth(tree, depth)

Return all node indices at a specified depth. 
"""
function find_at_depth(tree::BoxTree{N,T,I}, depth::Integer) where {N,T,I}
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

"""
    leaves(tree, initial_node_idx=1)

Return the node indices of all leaves. 
Begins search at `initial_node_idx`, i.e.
only returns node indices of nodes below 
`initial_node_idx` within the tree. 
"""
function leaves(tree::BoxTree{N,T,I}, initial_node_idx=1) where {N,T,I}
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

function Base.keys(tree::Q) where {N,T,I,Q<:BoxTree{N,T,I}}
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

"""
    hidden_keys(tree)

Return all keys within the tree, including 
keys not corresponding to leaf nodes. 
"""
function hidden_keys(tree::Q) where {N,T,I,Q<:BoxTree{N,T,I}}
    K = keytype(Q)
    keys = K[]
    queue = Tuple{I,K}[(1, K((1, ntuple(_->1,N))))]
    while !isempty(queue)
        node_idx, key = pop!(queue)
        node = tree.nodes[node_idx]
        push!(keys, key)
        if !isleaf(node)
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

function Base.:(==)(tr1::BoxTree{N,T,I}, tr2::BoxTree{N,V,J}) where {N,T,I,V,J}
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
