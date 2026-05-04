"""
Module utilitaire — primitives partagées par toutes les métaheuristiques.

Choix de conception clés :

1. **Matrice de distances précalculée** (DistMatrix) — Le coût dominant en TSP
   métaheuristique est l'évaluation répétée de la distance euclidienne (sqrt).
   On la calcule UNE FOIS au début de l'optimisation et on la passe à tous les
   algorithmes : ACO, GA, 2-opt et Hybrid en bénéficient simultanément.
   Gain : ~O(n²) sqrt évités par évaluation de tour, multiplié par des
   centaines de milliers d'évaluations.

2. **tour_distance** travaille désormais sur dist_matrix au lieu de coords —
   plus aucun appel à math.sqrt dans la boucle chaude.

3. **graph-aware** — si un graphe partiel est fourni, les arêtes interdites
   produisent +inf, ce qui fait rejeter automatiquement le tour invalide
   par toute fonction de sélection (min, sort, fitness).
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Set

# ── Types ─────────────────────────────────────────────────────────────────────
Tour = List[int]
Coords = List[Tuple[float, float]]
DistMatrix = List[List[float]]                  # dist[i][j] = euclidean(i, j)
Graph = Optional[Dict[int, Set[int]]]           # adjacence; None = graphe complet


# ── Distance et matrice ──────────────────────────────────────────────────────
def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance euclidienne 2D — utilisée uniquement pour BÂTIR la matrice."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(coords: Coords) -> DistMatrix:
    """
    Précalcule toutes les distances paire à paire en O(n²) sqrt.
    Symétrique : dist[i][j] == dist[j][i]. Diagonale à 0.0.

    Cette matrice remplace les appels répétés à euclidean_distance dans les
    boucles internes des métaheuristiques (gain de performance majeur).
    """
    n = len(coords)
    dist: DistMatrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist[i][j] = d
            dist[j][i] = d  # symétrie
    return dist


# ── Graphe ───────────────────────────────────────────────────────────────────
def build_graph(edge_list: List[Tuple[int, int]], n: int) -> Dict[int, Set[int]]:
    """
    Adjacence bidirectionnelle : (a, b) ⇒ a ∈ N(b) et b ∈ N(a).
    Sets pour lookup en O(1) lors du test d'arête autorisée.
    """
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edge_list:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def get_neighbors(city: int, graph: Graph, n: int) -> List[int]:
    """
    Voisins accessibles depuis `city`.
    Si graph est None → graphe complet (toutes les autres villes).
    """
    if graph is None:
        return [i for i in range(n) if i != city]
    return list(graph.get(city, set()))


def is_valid_edge(a: int, b: int, graph: Graph) -> bool:
    """Test O(1) : l'arête a→b est-elle autorisée ?"""
    if graph is None:
        return True
    return b in graph.get(a, set())


def is_valid_tour(tour: Tour, graph: Graph, n_cities: int) -> bool:
    """
    Garde-fou de validation finale : un tour est valide ssi
      • il visite chaque ville exactement une fois (permutation complète),
      • toutes ses arêtes (y compris la fermeture cyclique) sont
        autorisées par le graphe.

    Utilisé en post-traitement dans le router pour s'assurer qu'aucun
    algorithme ne renvoie silencieusement un tour invalide. Fail-fast.
    """
    if not tour or len(tour) != n_cities:
        return False
    if len(set(tour)) != n_cities:
        return False  # ville répétée
    if any(c < 0 or c >= n_cities for c in tour):
        return False  # index hors borne
    # Vérification arête par arête (incluant la fermeture cyclique)
    for i in range(n_cities):
        if not is_valid_edge(tour[i], tour[(i + 1) % n_cities], graph):
            return False
    return True


# ── Évaluation du tour ───────────────────────────────────────────────────────
def tour_distance(
    tour: Tour,
    dist_matrix: DistMatrix,
    graph: Graph = None,
) -> float:
    """
    Distance totale d'un tour cyclique en O(n) — sans aucun sqrt.

    Si `graph` est fourni et qu'une arête du tour n'y figure pas, retourne
    +inf : le tour est mathématiquement « infiniment mauvais » et sera donc
    écarté par tout opérateur de sélection (min, sort, tournament, etc.).
    """
    n = len(tour)
    if n < 2:
        return 0.0

    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]  # bouclage cyclique

        # Validation graphe partiel
        if graph is not None and b not in graph.get(a, set()):
            return float("inf")

        total += dist_matrix[a][b]

    return total


# ── Population aléatoire (graphe complet) ────────────────────────────────────
def random_population(size: int, n_cities: int) -> List[Tour]:
    """
    Population aléatoire pour graphe complet uniquement.
    Pour graphe partiel, utiliser `random_valid_population` du module genetic.
    """
    base = list(range(n_cities))
    population = []
    for _ in range(size):
        tour = base[:]
        random.shuffle(tour)
        population.append(tour)
    return population
