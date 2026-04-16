"""
algorithms/ant_colony.py
Algorithme ACO (Ant Colony Optimization) pour le TSP.

Principe :
  - Chaque fourmi construit un tour complet en choisissant
    la prochaine ville selon une probabilité combinant
    phéromones (mémoire collective) et visibilité (1/distance).
  - Après chaque itération, les phéromones s'évaporent
    et les fourmis déposent sur leurs arêtes (meilleures = plus de dépôt).
  - Après convergence, retourne le meilleur tour trouvé.
"""

import random
from typing import List, Tuple

from .utils import Tour, tour_distance, euclidean_distance


# ---------------------------------------------------------------------------
# Construction d'un tour par une fourmi
# ---------------------------------------------------------------------------

def _ant_tour(
    pheromones: List[List[float]],
    coords: List[Tuple[float, float]],
    alpha: float,
    beta: float,
) -> Tour:
    """
    Construit un tour complet pour une fourmi.

    À chaque étape, la fourmi choisit la prochaine ville
    selon la règle de transition probabiliste :

        P(i→j) ∝ τ(i,j)^alpha × η(i,j)^beta

    où τ = phéromone et η = 1/distance (visibilité).

    Args:
        pheromones: Matrice des phéromones [n×n].
        coords:     Coordonnées (x, y) des villes.
        alpha:      Poids des phéromones (exploitation).
        beta:       Poids de la visibilité (distance inverse).

    Returns:
        Tour complet : liste d'indices dans l'ordre de visite.
    """
    n = len(coords)
    start = random.randint(0, n - 1)
    visited = [False] * n
    visited[start] = True
    tour = [start]

    for _ in range(n - 1):
        current = tour[-1]

        # Calcul des scores pour chaque ville non visitée
        scores = []
        candidates = []
        for j in range(n):
            if visited[j]:
                continue
            dist = euclidean_distance(coords[current], coords[j])
            visibility = 1.0 / dist if dist > 0 else 1e10
            score = (pheromones[current][j] ** alpha) * (visibility ** beta)
            scores.append(score)
            candidates.append(j)

        # Sélection probabiliste (roulette wheel)
        total = sum(scores)
        if total == 0:
            next_city = random.choice(candidates)
        else:
            pick = random.uniform(0, total)
            cumulative = 0.0
            next_city = candidates[-1]  # fallback
            for city, score in zip(candidates, scores):
                cumulative += score
                if cumulative >= pick:
                    next_city = city
                    break

        visited[next_city] = True
        tour.append(next_city)

    return tour


# ---------------------------------------------------------------------------
# Mise à jour des phéromones
# ---------------------------------------------------------------------------

def _update_pheromones(
    pheromones: List[List[float]],
    tours: List[Tour],
    coords: List[Tuple[float, float]],
    evaporation: float,
    q: float,
) -> None:
    """
    Met à jour la matrice de phéromones (in-place) :
      1. Évaporation : toutes les phéromones diminuent.
      2. Dépôt       : chaque fourmi dépose Q / distance sur ses arêtes.

    Args:
        pheromones:  Matrice à mettre à jour.
        tours:       Tours construits par les fourmis cette itération.
        coords:      Coordonnées des villes.
        evaporation: Taux d'évaporation ρ ∈ ]0, 1[.
        q:           Constante de dépôt (plus Q est grand, plus le dépôt est fort).
    """
    n = len(pheromones)

    # Évaporation
    for i in range(n):
        for j in range(n):
            pheromones[i][j] *= (1.0 - evaporation)
            pheromones[i][j] = max(pheromones[i][j], 1e-6)  # seuil minimal

    # Dépôt proportionnel à la qualité du tour
    for tour in tours:
        dist = tour_distance(tour, coords)
        deposit = q / dist if dist > 0 else 0.0
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k + 1) % len(tour)]
            pheromones[a][b] += deposit
            pheromones[b][a] += deposit  # graphe non orienté


# ---------------------------------------------------------------------------
# Algorithme ACO principal
# ---------------------------------------------------------------------------

def run_ant_colony(
    coords: List[Tuple[float, float]],
    n_ants: int = 30,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.5,
    q: float = 100.0,
    initial_pheromone: float = 1.0,
) -> Tour:
    """
    Exécute l'algorithme ACO et retourne le meilleur tour trouvé.

    Args:
        coords:            Coordonnées (x, y) des villes.
        n_ants:            Nombre de fourmis par itération.
        iterations:        Nombre d'itérations.
        alpha:             Influence des phéromones.
        beta:              Influence de la visibilité (1/distance).
        evaporation:       Taux d'évaporation (ρ).
        q:                 Constante de dépôt de phéromones.
        initial_pheromone: Valeur initiale de toutes les phéromones.

    Returns:
        Meilleur tour trouvé (liste d'indices).
    """
    n = len(coords)

    # Initialisation de la matrice de phéromones
    pheromones: List[List[float]] = [
        [initial_pheromone] * n for _ in range(n)
    ]

    best_tour: Tour = []
    best_distance = float("inf")

    for _ in range(iterations):
        # Chaque fourmi construit son tour
        iteration_tours = [
            _ant_tour(pheromones, coords, alpha, beta)
            for _ in range(n_ants)
        ]

        # Mise à jour du meilleur global
        for tour in iteration_tours:
            dist = tour_distance(tour, coords)
            if dist < best_distance:
                best_distance = dist
                best_tour = tour[:]

        # Évaporation + dépôt (uniquement les meilleurs tours de l'itération)
        _update_pheromones(pheromones, iteration_tours, coords, evaporation, q)

    return best_tour