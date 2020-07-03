function relative_attractor(boxset::BoxSet, g::BoxMap, depth::Int)
    for k = 1:depth
        boxset = subdivide(boxset)
        boxset = g(boxset; target=boxset)
    end

    return boxset
end

function unstable_set!(boxset::BoxSet, g::BoxMap)
    boxset_new = boxset

    while !isempty(boxset_new)
        boxset_new = g(boxset_new)

        setdiff!(boxset_new, boxset)
        union!(boxset, boxset_new)
    end

    return boxset
end
