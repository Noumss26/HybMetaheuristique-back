# algorithms/utils.py

import random
import math
from typing import List, Tuple

# 🔥 Type Tour = liste d'indices
Tour = List[int]


# ---------------------------------------------------------------------------
# Distance entre deux points
# ---------------------------------------------------------------------------

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Distance totale d’un tour
# ---------------------------------------------------------------------------

def tour_distance(tour: Tour, coords: List[Tuple[float, float]]) -> float:
    if len(tour) < 2:
        return 0.0

    total = 0.0
    for i in range(len(tour)):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % len(tour)]]
        total += euclidean_distance(a, b)

    return total


# ---------------------------------------------------------------------------
# Générer une population aléatoire
# ---------------------------------------------------------------------------

def random_population(size: int, n_cities: int) -> List[Tour]:
    population = []
    base = list(range(n_cities))

    for _ in range(size):
        tour = base[:]
        random.shuffle(tour)
        population.append(tour)

    return population

# algorithms/utils.py

def is_valid_edge(a: int, b: int, graph: dict[int, list[int]]) -> bool:
    return b in graph.get(a, [])