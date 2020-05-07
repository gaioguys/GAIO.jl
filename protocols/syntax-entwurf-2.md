Syntaxentwurf – weitere Vorschläge
----------------------------------

# Grundlegende Struktur/Typen/Ideen

Folgende Ideen sind etwas sehr unsortiert (und lang geworden...) und recht allgemein gehalten (und könnten über das Ziel weit hinaus schießen); vermutlich kann man sonst überall, wo `Mesh` und `Cell` steht, stattdessen `BoxCollection` und `Box` o.ä. nehmen.

* `Mesh{T}`: ein unterteilbares Gitter. Könnte `AbstractSparseArray{Cell{T}, DIM}` implementieren... Aber: Wahrscheinlich besser, wenn nur Iterator...
    * Stellt nur die am meisten unterteilteste Ebene des Baumes dar.
    * Vermutlich einzige implementierte Realisierung: `BoxCollection{T}` oder ähnlicher Name
    * Implementierungsidee: Ähnlich wie `SparseMatrix`, nur mehrdimensional? (entspricht praktisch dem jetzigen Vorschlag) Vielleicht Pyramidenartig... Für jedes Level eine `SparseMatrix`-artige Implementierung... (welche man vorerst vielleicht sogar als Julia-Set oder vielleicht als in-memory binary heap o.ä. tun könnte – oder einfach die Liste sortieren bei Bedarf... Könnte man lazy tun, also nur dann sortieren, was gebraucht wird und danach immer nur den neuen Teil der Liste sortieren und dann mergen) Würde insgesamt nicht so sehr auf 64 Bit beschränken, sondern hier variabler sein und dem JIT-Compiler die Optimierung überlassen (also vorerst zumindest eher abstraktere Indizes formulieren)
    * `SubMesh{T}` analog zu `SubArray{T}`; implementiert auch `AbstractSet{Cell{T}}`
        * Generelle Idee: `Mesh{T}` als Grundmenge, `SubMesh{T}` als Untermengen, die entsprechend verarbeitet werden können. Aber nur `Mesh{T}` hält die Daten
        * `Mesh{T}` ist selber ein `SubMesh{T}`
        * Implementierungsidee: Hauptsächlich auf Iteratoren setzen (die lazy entfaltet werden) und KEINE Kopien von Indizes anfertigen (so haben Testfunktionen auch noch eine Berechtigung)
        * `SubMesh{T}`es sind an das zugehörige Mesh gebunden, ähnlich wie SubArrays
        * `SubMesh{T}` ist immutable
        * Testfunktionen bilden auch ein `SubMesh{T}`
    * `subdivide!(::SubMesh{T})`: Fügt den ausgewählten Zellen eine Unterteilungsstufe hinzu
        * Hat zur Folge: Jegliche `SubMesh{T}`es iterieren ab sofort über alle unterteilten Zellen, die sie vorher enthalten haben; automatisch... Auch für das eingegebene `SubMesh{T}`
        * Thread-safety?
        * `subdivide_until!(::SubMesh{T}, ::Function{(::Cell) -> Bool})` unterteilt alle Zellen und Kindzellen, die `true` zurückgeben
    * für Test-Funktionen: `test(::SubMesh{T}, func::Function{Decide_per_cell}) -> ::SubMesh{T}`:
        * Entscheidet pro Zelle, ob diese im `::SubMesh{T}` behalten werden soll
    * `find_cell_on_level!(::Vector{T}, level::Integer) -> Cell{T}` (Name kann gerne verändert werden) gibt Zelle auf Ebene `level` zurück, unterteilt evtl. (evtl. fusionieren mit `extend`); Variante `find_cell!(::Vector{T}) -> Cell{T}` sucht die Zelle nur
    * vielleicht gar kein `remove`?
    * Denkbar: `extend` Nur als potenzielle Idee, wäre sowieso bei allgemeinen Meshes auch schwer zu implementieren... Zum Beispiel wäre denkbar: `extend(position::Vector)`, sodass eine Box der Startgröße erzeugt wird, sodass `position` überdeckt ist. Vielleicht auch `extend(position::Vector, level::Integer)`, welche gleich soweit um `position` unterteilt, dass eine Box, die `position` enthält existiert und auf Subdivision-Level `level` liegt
        * Praktisch: Das Starter-Mesh ist auch nur eine Zelle eines evtl. größeren Meshes
* `Cell{T}`:
    * Ist selber ein `SubMesh{T}` mit nur einer Zelle enthalten (kann folglich auch alleine unterteilt werden)
    * Hat Informationen über Zentrum, Ecken, Level, etc. (bleibt so auch erhalten, wenn unterteilt; kann aber danach nicht mehr aufgerufen werden)
    * Implementiert Methoden wie `contains(::Cell{T}, ::Vector{T})` usw.
    * Zusätzlich noch: `T` als Payload-generischen Parameter (wird auf alle Unterzellen kopiert beim Unterteilen)
* `function_map(::SubMesh{T}, func::FUNCTION_TYPE) -> ::SubMesh{T}`: Gedacht als Typ, der einer Funktion entspricht
    * zum Anwenden einer Funktion auf das `SubMesh{T}`, gibt eine Untermenge der bestehenden Cells im `Mesh{T}` zurück (d.h. implementiert effektiv das „es gibt ein B', sodass f(B') \cap B nichtleer ist“, wobei alle Bs und B's als aus dem Mesh entnommen betrachtet werden)... Alternativ auch einen `IntersectionTester` als generischen Typen auf das Mesh erlauben, um so dann `func(mesh)` schreiben zu können
        * Beispiel: `gridpoint_intersecter(num_points::Integer) -> ::Function{::SubMesh{T}} -> ::SubMesh{T}`
        * Zusätzlich für Punkte-Tester: `function_map_count(::SubMesh{::Integer}, func::FUNCTION_TYPE, point_count::Integer) -> ::SubMesh{::Integer}`; zählt, wie viele Punkte was getroffen haben
* Allgemeine Methoden:
    * Natürlich: Gängige Algorithmen vor-implementiert
    * Vielleicht (nur vielleicht...) auch kleine Helfermethoden wie
        * `iterative_subdivision(start::SubMesh{T}, n::Integer, ::Function{(::Integer, ::SubMesh{T}) -> ::SubMesh{T}}) -> ::Mesh`: initialisiert ein neues Mesh und iteriert n-mal und unterteilt jedes Mal das verbleibende Mesh (vielleicht noch mit Mesh als Eingabeparameter), gibt das Ergebnis zurück
        * Oder auch: Zellen-zentrierte Algorithmen... Siehe Beispiel

# Offene Fragen

* Erlauben wir Zugriff auf bereits zerteilte Zellen?

# Beispiele
(vermutlich alles andere als Julia-konform...)

## Subdivision-Algorithmus
```
# Erläuterung: gridpoint_mapper soll hier eine Funktion zurückgeben (und eigentlich vermutlich besser vor dem Befehl definiert werden...)
# (Parameter der BoxCollection hier weggelassen)
# Der Code hier ist noch nicht 100%ig optimal, 

mapper = gridpoint_mapper(16, SAME_LEVEL)

subdiv = iterative_subdivision(BoxCollection(), 10,
    (_, submesh)
        -> intersect(mapper(submesh, f), submesh)
)

# oder:

mesh = BoxCollection()
subdiv = mesh # evtl. subdiv = submesh(mesh)
for i = 1:10
    subdiv = intersect(mapper(submesh, f), submesh)
    subdivide!(subdiv)
end

# TODO: den SAME_LEVEL-Parameter auf das SubMesh verschieben
```

## Continuation-Algorithmus
```
# nehme subdiv von vorheriger Ausführung

mapper2 = gridpoint_mapper(16, SUBDIVIDE(10))
subview = subdiv
for i = 1:n
    subview = mapper2(subview, f)
end

# TODO: den SUBDIVIDE(10)-Parameter auf das SubMesh verschieben (meint: immer nur auf Cells in Ebene 10 mappen, nicht auf die gleiche Ebene, wie bei SAME_LEVEL)
# Z.B.: lazy_subdivide(subdiv, 10) führt gibt neues SubMesh zurück, das genauso wie das vorherige ist, nur entsprechend mehr unterteilt
```

## Subdivision Alternativ (vermutlich ungeeignet und nicht wirklich in Julia geschrieben)
```
# Experimentelle Idee... Viel von dem hier ist noch nicht oben beschrieben
# Der Gedanke ist: Die Algorithmen nur über Operationen auf Zellen zu beschreiben, nicht auf die Ebenen
# Hier: Mesh == BoxCollection
# Hat momentan noch Fehler...

hit_collector = # Datenstruktur, die pro Ebene aufzeichnet, ob eine Zelle irgendwann mal getroffen wurde, selbst wenn sie noch nicht existiert (Problem: Speicherverbrauch...)

# Idee hier: Verarbeite eine Zelle, gebe neue Zellen zum Verarbeiten zurück
function cell_op(cell::Cell{Bool}, mesh::Mesh{Payload})
    if level(cell) < 10 then
        cells_hit = gridpoint_mapper(16, SUBDIVIDE(level(cell)))(cell, f)
        new_cells = []
        for c in cells_hit do
            if cell.data and check_and_subdivide(cell) then # check_and_subdivide unterteilt eine Zelle, falls noch nicht geschehen und gibt nur dann true zurück
                new_cells += children(c)
            end
        end
        if hit(hit_collector, cell) and check_and_subdivide(cell) then
            subdivide!(cell)
            new_cells += children(cell)
        else
            cell.data = True # oder irgendwie so...
        end
        return new_cells
    else
        return []
    end
end

# cell_operation wird für alle Zellen in der Eingabe aufgerufen und führt dann obige Funktion aus -- für alle diese Zellen und alle, die von da zurückgegeben werden usw.
subdiv = cell_operation(::BoxCollection(), cell_op)
```

