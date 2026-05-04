"""
Ant Colony Optimization (ACO) — version professionnelle.

Variante implémentée : **Ant System** (Dorigo, Maniezzo & Colorni 1996).

Optimisations clés vs implémentation naïve :

1. **Matrice de visibilité η précalculée**
   La visibilité η_ij = 1/d_ij est constante au long du run. La pré-élever
   à la puissance β nous épargne β-power à chaque pas de chaque fourmi
   (~n × n_ants × iterations exponentiations économisées).

2. **Matrice de distances partagée** — plus aucun sqrt dans la boucle.

3. **Pré-calcul du tour-cost via matrice** — cohérent avec le reste du code.

4. **Construction du tour graph-aware** : les fourmis ne peuvent emprunter
   que des arêtes autorisées par le graphe partiel ; en cas d'impasse,
   le tour est rejeté plutôt que de retourner un tour invalide.

Hyperparamètres par défaut (Dorigo) : α=1.0, β=2.0, ρ=0.5, Q=100, 30
fourmis, 100 itérations. Convient à des instances ≤ ~50 villes.
"""

import random
from typing import List

from .utils import Tour, DistMatrix, Graph, get_neighbors, tour_distance


# ─────────────────────────────────────────────────────────────────────────────
# Construction d'un tour par une fourmi
# ─────────────────────────────────────────────────────────────────────────────
def _ant_tour(
    pheromones: List[List[float]],
    heuristic_pow: List[List[float]],   # η^β précalculé
    n: int,
    graph: Graph,
    alpha: float,
) -> Tour:
    """
    Construit un tour pour une fourmi en respectant la topologie.

    Règle de transition (probabiliste) :
        p(j | i) ∝ τ_ij^α · η_ij^β

    où τ = phéromones, η = visibilité = 1/distance.
    Sélection par roulette pondérée parmi les voisins non visités.

    Retourne [] si la fourmi se retrouve en impasse (graphe trop contraint
    ou retour au point de départ impossible).
    """
    start = random.randint(0, n - 1)
    visited = [False] * n
    visited[start] = True
    tour: Tour = [start]

    for _ in range(n - 1):
        current = tour[-1]

        # Voisins ATORISÉS par le graphe ET non encore visités
        neighbors = [j for j in get_neighbors(current, graph, n) if not visited[j]]
        if not neighbors:
            return []  # impasse — tour partiel rejeté

        # Calcul des scores τ^α · η^β pour chaque voisin
        scores = []
        total = 0.0
        for j in neighbors:
            score = (pheromones[current][j] ** alpha) * heuristic_pow[current][j]
            scores.append(score)
            total += score

        # Roulette wheel — fallback uniform si tous scores nuls (rare)
        if total <= 0.0:
            next_city = random.choice(neighbors)
        else:
            pick = random.random() * total
            cumulative = 0.0
            next_city = neighbors[-1]  # filet de sécurité numérique
            for j, score in zip(neighbors, scores):
                cumulative += score
                if cumulative >= pick:
                    next_city = j
                    break

        visited[next_city] = True
        tour.append(next_city)

    # Vérification de la fermeture du cycle (retour au start)
    if graph is not None and start not in graph.get(tour[-1], set()):
        return []

    return tour


# ─────────────────────────────────────────────────────────────────────────────
# Mise à jour des phéromones (évaporation + dépôt)
# ─────────────────────────────────────────────────────────────────────────────
def _update_pheromones(
    pheromones: List[List[float]],
    tours: List[Tour],
    dist_matrix: DistMatrix,
    graph: Graph,
    evaporation: float,
    q: float,
    pheromone_min: float = 1e-6,
) -> None:
    """
    Mise à jour Ant System :
      τ_ij ← (1−ρ) · τ_ij + Σ_k Δτ_ij^k

    où Δτ_ij^k = Q / L^k si la fourmi k a emprunté l'arête (i,j), 0 sinon.

    Le seuil pheromone_min évite τ → 0 (perte définitive d'arête).
    """
    n = len(pheromones)

    # Évaporation uniforme avec floor
    decay = 1.0 - evaporation
    for i in range(n):
        row = pheromones[i]
        for j in range(n):
            v = row[j] * decay
            row[j] = v if v > pheromone_min else pheromone_min

    # Dépôt par les fourmis ayant produit un tour valide
    for tour in tours:
        L = tour_distance(tour, dist_matrix, graph)
        if L == float("inf") or L == 0.0:
            continue
        deposit = q / L
        m = len(tour)
        for k in range(m):
            a = tour[k]
            b = tour[(k + 1) % m]
            pheromones[a][b] += deposit
            pheromones[b][a] += deposit  # symétrique


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale ACO
# ─────────────────────────────────────────────────────────────────────────────
def run_ant_colony(
    dist_matrix: DistMatrix,
    n_cities: int,
    graph: Graph = None,
    n_ants: int = 30,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.5,
    q: float = 100.0,
    initial_pheromone: float = 1.0,
) -> Tour:
    """
    Ant System classique avec visibilité précalculée.

    Args:
        dist_matrix:  matrice n×n des distances euclidiennes
        n_cities:     nombre de villes (= len(dist_matrix))
        graph:        adjacence partielle (None = graphe complet)
        n_ants:       nombre de fourmis par itération
        iterations:   nombre d'itérations (générations de fourmis)
        alpha, beta:  poids relatifs phéromone/visibilité
        evaporation:  taux d'évaporation ρ ∈ ]0, 1[
        q:            constante de dépôt Q

    Returns:
        meilleur tour trouvé sur l'ensemble des itérations.
    """
    n = n_cities

    # ── Pré-calcul de η^β ────────────────────────────────────────────
    # Constant pour toute la durée du run, économise des centaines de
    # milliers d'exponentiations.
    HUGE = 1e10
    heuristic_pow: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist_matrix[i][j]
            eta = (1.0 / d) if d > 0 else HUGE
            heuristic_pow[i][j] = eta ** beta

    # ── Phéromones initiales uniformes ───────────────────────────────
    pheromones = [[initial_pheromone] * n for _ in range(n)]

    # ── Mémoire du meilleur tour global ──────────────────────────────
    best_tour: Tour = []
    best_distance = float("inf")

    for _ in range(iterations):
        # Construction parallèle (logique) de n_ants tours
        valid_tours: List[Tour] = []
        for _ in range(n_ants):
            tour = _ant_tour(pheromones, heuristic_pow, n, graph, alpha)
            if tour:
                valid_tours.append(tour)

        # Mise à jour du best avec les tours de cette itération
        for tour in valid_tours:
            d = tour_distance(tour, dist_matrix, graph)
            if d < best_distance:
                best_distance = d
                best_tour = tour[:]

        # Évaporation + dépôt (skip si aucun tour valide → évite décroissance pure)
        if valid_tours:
            _update_pheromones(
                pheromones, valid_tours, dist_matrix, graph, evaporation, q
            )

    return best_tour
