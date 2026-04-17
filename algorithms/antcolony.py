import random
from typing import List, Tuple, Optional

from .utils import Tour, tour_distance, euclidean_distance, Graph, get_neighbors


def _ant_tour(
    pheromones: List[List[float]],
    coords: List[Tuple[float, float]],
    alpha: float,
    beta: float,
    graph: Graph = None,
) -> Optional[Tour]:
    """
    Construit un tour pour une fourmi en respectant la topologie du graphe.
    Retourne None si aucun tour complet n'est possible depuis le point de départ
    (graphe non connexe ou trop contraint).
    """
    n = len(coords)
    start = random.randint(0, n - 1)
    visited = [False] * n
    visited[start] = True
    tour = [start]

    for _ in range(n - 1):
        current = tour[-1]

        # Voisins autorisés par le graphe ET non encore visités
        neighbors = [
            j for j in get_neighbors(current, graph, n)
            if not visited[j]
        ]

        if not neighbors:
            # Impasse : impossible de compléter le tour
            return None

        # Scores probabilistes τ^alpha × η^beta
        scores = []
        for j in neighbors:
            dist = euclidean_distance(coords[current], coords[j])
            visibility = 1.0 / dist if dist > 0 else 1e10
            score = (pheromones[current][j] ** alpha) * (visibility ** beta)
            scores.append(score)

        # Sélection par roulette
        total = sum(scores)
        if total == 0:
            next_city = random.choice(neighbors)
        else:
            pick = random.uniform(0, total)
            cumulative = 0.0
            next_city = neighbors[-1]
            for city, score in zip(neighbors, scores):
                cumulative += score
                if cumulative >= pick:
                    next_city = city
                    break

        visited[next_city] = True
        tour.append(next_city)

    # Vérifier que le retour au départ est possible
    if graph is not None and start not in graph.get(tour[-1], set()):
        return None  # la boucle de retour n'existe pas

    return tour


def _update_pheromones(
    pheromones: List[List[float]],
    tours: List[Tour],
    coords: List[Tuple[float, float]],
    evaporation: float,
    q: float,
    graph: Graph = None,
) -> None:
    n = len(pheromones)

    for i in range(n):
        for j in range(n):
            pheromones[i][j] *= (1.0 - evaporation)
            pheromones[i][j] = max(pheromones[i][j], 1e-6)

    for tour in tours:
        dist = tour_distance(tour, coords, graph)
        if dist == float("inf") or dist == 0:
            continue
        deposit = q / dist
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k + 1) % len(tour)]
            pheromones[a][b] += deposit
            pheromones[b][a] += deposit


def run_ant_colony(
    coords: List[Tuple[float, float]],
    graph: Graph = None,
    n_ants: int = 30,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.5,
    q: float = 100.0,
    initial_pheromone: float = 1.0,
) -> Tour:
    n = len(coords)
    pheromones = [[initial_pheromone] * n for _ in range(n)]

    best_tour: Tour = []
    best_distance = float("inf")

    for _ in range(iterations):
        iteration_tours = []
        for _ in range(n_ants):
            tour = _ant_tour(pheromones, coords, alpha, beta, graph)
            if tour is not None:          # ignorer les tours invalides
                iteration_tours.append(tour)

        for tour in iteration_tours:
            dist = tour_distance(tour, coords, graph)
            if dist < best_distance:
                best_distance = dist
                best_tour = tour[:]

        if iteration_tours:
            _update_pheromones(pheromones, iteration_tours, coords, evaporation, q, graph)

    return best_tour