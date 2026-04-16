"""
algorithms/local_search.py
Implémentation du 2-opt : heuristique de recherche locale
qui améliore un tour en supprimant les croisements d'arêtes.
"""

from typing import List, Tuple

from .utils import Tour, tour_distance


# ---------------------------------------------------------------------------
# Opération 2-opt
# ---------------------------------------------------------------------------

def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
    """
    Effectue un swap 2-opt : inverse le segment [i+1 .. k].

    Exemple :
        tour   = [A, B, C, D, E]   i=1, k=3
        résultat = [A, B, D, C, E]  (segment B→D inversé)

    Args:
        tour: Tour courant.
        i:    Indice de début du segment (exclusif).
        k:    Indice de fin du segment (inclusif).

    Returns:
        Nouveau tour avec le segment inversé.
    """
    return tour[:i + 1] + tour[i + 1:k + 1][::-1] + tour[k + 1:]


# ---------------------------------------------------------------------------
# 2-opt complet
# ---------------------------------------------------------------------------

def two_opt(
    tour: Tour,
    coords: List[Tuple[float, float]],
    max_iterations: int = 1000,
) -> Tour:
    """
    Applique le 2-opt jusqu'à convergence (ou limite d'itérations).

    Principe :
      Pour chaque paire (i, k), tenter d'inverser le segment.
      Si la distance diminue → accepter le swap et repartir.
      Continuer jusqu'à ce qu'aucun swap n'améliore la solution.

    Args:
        tour:           Tour initial (liste d'indices).
        coords:         Coordonnées (x, y) des villes.
        max_iterations: Sécurité anti-boucle infinie.

    Returns:
        Tour amélioré.
    """
    best_tour = tour[:]
    best_distance = tour_distance(best_tour, coords)
    n = len(best_tour)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 1):
            for k in range(i + 2, n):
                # Ne pas tester le segment qui boucle (i=0, k=n-1)
                if i == 0 and k == n - 1:
                    continue

                candidate = two_opt_swap(best_tour, i, k)
                candidate_distance = tour_distance(candidate, coords)

                if candidate_distance < best_distance - 1e-10:
                    best_tour = candidate
                    best_distance = candidate_distance
                    improved = True
                    break  # recommencer depuis le début (first improvement)

            if improved:
                break

    return best_tour