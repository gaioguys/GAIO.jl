struct BoxList{P<:BoxPartition,L<:AbstractVector}
    partition::P
    keylist::L
end

Base.length(list::BoxList) = length(list.keylist)
Base.getindex(list::BoxList, x::AbstractVector) = BoxList(list.partition, list.keylist[x])

BoxList(set::BoxSet) = BoxList(set.partition, collect(set.set))

BoxSet(list::BoxList) = BoxSet(list.partition, Set(list.keylist))
