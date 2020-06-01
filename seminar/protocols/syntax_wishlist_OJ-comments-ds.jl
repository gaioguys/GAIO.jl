#Algorithmen sind schön kurz (genug)

function relative_attractor!(B::BoxSet, f::DynamicalSystem, depth::Int)
# Soll es BoxSet heißen? Würde mir prinzipiell passen...
# Sollte dann nach Möglichkeit AbstractSet und vielleicht AbstractArray (wenn wir die Indizierung irgendwie brauchbar hinbekommen), dann als Union-Type.
# Und evtl. BoxSet{T}?
# Und allgemein noch andere Fragen/Kommentare zu meinem vorherigen Entwurf...
# Ist BoxSet die Datenstruktur selber oder nur eine View darauf? Wobei: Für eigentliche Benutzung GRÖẞTENTEILS unerheblich
# In Theorie könnten wir auch die Unterscheidung BoxSet und View auf BoxSet komplett intern machen und bei Bedarf neue Datenstrukturen etc. erzeugen...
# Müsste unsere Library dann nur von selber entscheiden, ob es eine View oder eine Kopie erstellen sollte...

# attractor relative to the box set B up to the given depth
    for k = 1:depth
        B = subdivide(B)
        B = intersect(B,f(B))   # or f.(B) ? <-- ja, lieber das; wir testen ja für jede Zelle (wir müssen dann aber das Broadcasting etwas verändern)
        # Allgemein: Soll das Testen auf Überschneiden wirklich so weit verborgen werden?
        # Können wir schon machen, dass stets ein Default-Tester ausgewählt wird
        # (würde ich dann aber tendenziell eher global oder an Problem und nicht an Mesh gebunden machen...)
        # Aber: Ich würde trotzdem noch Funktionen anbieten, sodass man die Testvariante noch auswählen kann
        # z.B. der Form function_map(f, B)
    end
end

function unstable_set!(B::BoxSet, f::DynamicalSystem)
# unstable set of the box set B
    C = B
    fB = f(B) # lieber f.(B), siehe oben
    while fB ≠ ∅             # or !isempty(fB) ... vllt. beides?
        B = setdiff(fB,C) # passt
        C = union(C,fB) # passt
        fB = f(B) # lieber f.(B), siehe oben
    end
end # <-- fehlte

function transition_matrix!(B::BoxSet, f::DynamicalSystem)
# topological transition matrix between the boxes in B
    L = Lipschitz(f)   # estimates the Lipschitz constant of f # <-- gehört aber nicht zu unserem Paket, oder?
    n = size(B) # passt
    E = zeros(n*L,3)
    for b in B
        fb = f(b), m = size(fb) # hier vielleicht eben f.(b) und f(b) möglich...?
        append!(E, [ones(m)*no(b), no.(fb), ones(m)]) # no(b) is in 1:n
    end
    I, J, V = E[1], E[2], E[3]
    T = sparse(I, J, V, n, n, combine)
end # <-- fehlte wieder
