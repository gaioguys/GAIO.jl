
function relative_attractor!(B::BoxSet, f::DynamicalSystem, depth::Int)
# attractor relative to the box set B up to the given depth
    for k = 1:depth
        B = subdivide(B)
        B = intersect(B,f(B))   # or f.(B) ?
    end
end

function unstable_set!(B::BoxSet, f::DynamicalSystem)
# unstable set of the box set B
C = B
fB = f(B)
while fB ≠ ∅             # or !isempty(fB)
    B = setdiff(fB,C)
    C = union(C,fB)
    fB = f(B)
end

function transition_matrix!(B::BoxSet, f::DynamicalSystem)
# topological transition matrix between the boxes in B
L = Lipschitz(f)   # estimates the Lipschitz constant of f
n = size(B)
E = zeros(n*L,3)
for b in B
    fb = f(b), m = size(fb)
    append!(E, [ones(m)*no(b), no.(fb), ones(m)]) # no(b) is in 1:n
end
I, J, V = E[1], E[2], E[3]
T = sparse(I, J, V, n, n, combine)
