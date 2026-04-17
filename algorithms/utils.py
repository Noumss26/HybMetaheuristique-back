import random
import math
from typing import List, Tuple, Optional, Dict, Set

Tour = List[int]

# Graphe : adjacency set, index → set d'indices voisins
Graph = Optional[Dict[int, Set[int]]]


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_graph(
    edge_list: List[Tuple[int, int]],
    n: int,
) -> Dict[int, Set[int]]:
    """
    Construit un dictionnaire d'adjacence à partir d'une liste d'arêtes.
    Chaque arête est bidirectionnelle : (a, b) → a peut aller en b ET b peut aller en a.

    Args:
        edge_list: liste de (index_ville_a, index_ville_b)
        n:         nombre total de villes

    Returns:
        Dict[int, Set[int]] — voisins accessibles depuis chaque ville
    """
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edge_list:
        graph[a].add(b)
        graph[b].add(a)  # bidirectionnel
    return graph


def get_neighbors(city: int, graph: Graph, n: int) -> List[int]:
    """
    Retourne les voisins accessibles depuis `city`.
    Si graph est None (graphe complet), retourne toutes les autres villes.
    """
    if graph is None:
        return [i for i in range(n) if i != city]
    return list(graph.get(city, set()))


def tour_distance(
    tour: Tour,
    coords: List[Tuple[float, float]],
    graph: Graph = None,
) -> float:
    """
    Calcule la distance totale d'un tour.
    Si graph fourni, retourne inf si une arête n'est pas dans le graphe
    (tour invalide selon la topologie).
    """
    if len(tour) < 2:
        return 0.0

    total = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]

        if graph is not None and b not in graph.get(a, set()):
            return float("inf")  # arête interdite

        total += euclidean_distance(coords[a], coords[b])

    return total


def random_population(size: int, n_cities: int) -> List[Tour]:
    base = list(range(n_cities))
    population = []
    for _ in range(size):
        tour = base[:]
        random.shuffle(tour)
        population.append(tour)
    return population


def is_valid_edge(a: int, b: int, graph: Graph) -> bool:
    """Vérifie si l'arête a→b est autorisée dans le graphe."""
    if graph is None:
        return True
    return b in graph.get(a, set())