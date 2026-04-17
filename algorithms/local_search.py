from typing import List, Tuple
from .utils import Tour, tour_distance, Graph


def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
    return tour[:i + 1] + tour[i + 1:k + 1][::-1] + tour[k + 1:]


def two_opt(
    tour: Tour,
    coords: List[Tuple[float, float]],
    max_iterations: int = 1000,
    graph: Graph = None,
) -> Tour:
    """
    2-opt qui respecte la topologie du graphe :
    un swap n'est accepté que si le tour résultant
    ne contient aucune arête interdite.
    """
    best_tour = tour[:]
    best_distance = tour_distance(best_tour, coords, graph)
    n = len(best_tour)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 1):
            for k in range(i + 2, n):
                if i == 0 and k == n - 1:
                    continue

                candidate = two_opt_swap(best_tour, i, k)
                candidate_distance = tour_distance(candidate, coords, graph)

                # Rejeté automatiquement si une arête est hors graphe (inf)
                if candidate_distance < best_distance - 1e-10:
                    best_tour = candidate
                    best_distance = candidate_distance
                    improved = True
                    break

            if improved:
                break

    return best_tour