"""
Router d'optimisation — endpoint POST /optimize.

Orchestration des 4 algorithmes (ACO, GA, 2-opt, Hybrid) avec :

1. **Matrice de distances calculée UNE SEULE fois par requête** et partagée
   entre tous les algorithmes — élimine totalement les sqrt redondants
   à travers le pipeline.

2. **Random baseline cohérente** pour la métrique improvement_percent.

3. **Tolérance aux pannes** : si un algorithme échoue, on continue avec
   les autres et on collecte les erreurs ; échec total → HTTP 500 explicite.
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from models.city import OptimizeRequest, OptimizeResponse, AlgoBreakdown

from algorithms.antcolony import run_ant_colony
from algorithms.genetic import run_genetic_algorithm, random_valid_population
from algorithms.hybrid import run_hybrid_optimization
from algorithms.local_search import two_opt
from algorithms.utils import (
    build_graph,
    build_distance_matrix,
    tour_distance,
    is_valid_tour,
)

logger = logging.getLogger("optimizer")
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/optimize", tags=["Optimisation"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _round_distance(d: float) -> float:
    """Arrondi à 4 décimales pour la sérialisation JSON."""
    return round(d, 4) if d != float("inf") else d


def _rotate_to_start(tour, start_index):
    """
    Aligne le tour pour qu'il commence par `start_index` (rotation cyclique).
    No-op si start_index n'est pas dans le tour.
    """
    if start_index is None or start_index not in tour:
        return tour
    idx = tour.index(start_index)
    return tour[idx:] + tour[:idx]


# ─────────────────────────────────────────────────────────────────────────────
# Wrappers — chaque algo reçoit la dist_matrix précalculée
# ─────────────────────────────────────────────────────────────────────────────
def _run_antcolony(dist_matrix, n, graph):
    tour = run_ant_colony(dist_matrix=dist_matrix, n_cities=n, graph=graph)
    return tour, tour_distance(tour, dist_matrix, graph) if tour else float("inf")


def _run_genetic(dist_matrix, n, graph):
    tour = run_genetic_algorithm(dist_matrix=dist_matrix, n_cities=n, graph=graph)
    return tour, tour_distance(tour, dist_matrix, graph) if tour else float("inf")


def _run_hybrid(dist_matrix, n, graph):
    return run_hybrid_optimization(dist_matrix=dist_matrix, n_cities=n, graph=graph)


def _run_local_search(dist_matrix, n, graph):
    """
    2-opt seul a besoin d'un point de départ : on tire un tour valide
    aléatoire, puis on l'optimise. Représente la « recherche locale pure ».
    """
    seeds = random_valid_population(1, n, graph)
    if not seeds:
        return [], float("inf")
    initial = seeds[0]
    tour = two_opt(initial, dist_matrix=dist_matrix, graph=graph)
    return tour, tour_distance(tour, dist_matrix, graph)


ALGORITHMS = {
    "antcolony": _run_antcolony,
    "genetic": _run_genetic,
    "hybrid": _run_hybrid,
    "local_search": _run_local_search,
}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint principal
# ─────────────────────────────────────────────────────────────────────────────
@router.post("", response_model=OptimizeResponse)
async def optimize(payload: OptimizeRequest) -> OptimizeResponse:
    coords = [(c.x, c.y) for c in payload.cities]
    names = [c.name for c in payload.cities]
    n = len(coords)

    if n < 2:
        raise HTTPException(status_code=400, detail="Au moins 2 villes requises.")

    # ── Pré-calculs partagés (UNE SEULE fois par requête) ───────────
    dist_matrix = build_distance_matrix(coords)

    graph = None
    if payload.edges:
        edge_indices = []
        for edge in payload.edges:
            if edge.city_a not in names or edge.city_b not in names:
                raise HTTPException(status_code=400, detail="Arête invalide")
            edge_indices.append((names.index(edge.city_a), names.index(edge.city_b)))
        graph = build_graph(edge_indices, n)
        logger.info("🗺️  Graphe partiel : %d arêtes", len(edge_indices))
    else:
        logger.info("🗺️  Graphe complet")

    # ── Ville de départ (optionnelle) ────────────────────────────────
    start_index = None
    if payload.start_city:
        if payload.start_city not in names:
            raise HTTPException(status_code=400, detail="start_city invalide")
        start_index = names.index(payload.start_city)

    # ─────────────────────────────────────────────────────────────────
    # Lancement de TOUS les algorithmes pour comparaison empirique
    # (mode benchmark — c'est l'objectif académique du projet ADOMC)
    # ─────────────────────────────────────────────────────────────────
    results = []
    errors: dict = {}
    t_start = time.perf_counter()

    # On mesure le temps PAR algorithme pour le breakdown affiché au front
    breakdown: list = []

    for name, algo in ALGORITHMS.items():
        algo_t0 = time.perf_counter()
        try:
            logger.info("⚙️  Test : %s", name)
            tour, dist = algo(dist_matrix, n, graph)

            # ── Garde-fou : on rejette tout tour invalide ───────────
            if not is_valid_tour(tour, graph, n):
                raise ValueError(
                    "Tour invalide : arête interdite par le graphe ou "
                    "permutation incomplète."
                )
            if dist == float("inf"):
                raise ValueError("Distance infinie (arête interdite).")

            algo_ms = round((time.perf_counter() - algo_t0) * 1000, 2)
            results.append({"name": name, "tour": tour, "distance": dist})
            breakdown.append({"name": name, "distance": _round_distance(dist), "time_ms": algo_ms})
        except Exception as e:
            logger.warning("❌ %s : %s", name, e)
            errors[name] = str(e)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    if not results:
        raise HTTPException(
            status_code=500,
            detail={"message": "Tous les algorithmes ont échoué", "errors": errors},
        )

    # ── Sélection du meilleur tour, toutes méthodes confondues ──────
    best = min(results, key=lambda r: r["distance"])
    tour_indices = _rotate_to_start(best["tour"], start_index)

    optimal_path = [names[i] for i in tour_indices]
    optimal_distance = _round_distance(best["distance"])
    algorithm_used = best["name"]

    # ── Baseline aléatoire pour calcul de l'amélioration ────────────
    # On ne génère une baseline QUE si on peut produire un tour
    # graph-aware valide. Sinon on n'affiche pas de baseline bidon
    # (l'ancienne version retombait sur list(range(n)) qui pouvait
    # contenir des arêtes interdites — bug visuel).
    rand_seeds = random_valid_population(1, n, graph)
    if rand_seeds and is_valid_tour(rand_seeds[0], graph, n):
        random_indices = _rotate_to_start(rand_seeds[0], start_index)
        random_path = [names[i] for i in random_indices]
        random_distance = _round_distance(
            tour_distance(random_indices, dist_matrix, graph)
        )
        if random_distance > 0 and random_distance != float("inf"):
            improvement = round(
                ((random_distance - optimal_distance) / random_distance) * 100, 2
            )
        else:
            improvement = 0.0
    else:
        # Pas de baseline disponible (graphe contraint, peu de hamiltoniens)
        random_path = []
        random_distance = 0.0
        improvement = 0.0

    return OptimizeResponse(
        optimal_path=optimal_path,
        total_distance=optimal_distance,
        algorithm_used=algorithm_used,
        execution_time_ms=elapsed_ms,
        random_path=random_path,
        random_distance=random_distance,
        improvement_percent=improvement,
        start_city=payload.start_city,
        breakdown=[AlgoBreakdown(**b) for b in breakdown],
    )


# ─────────────────────────────────────────────────────────────────────────────
@router.get("/algorithms")
async def list_algorithms():
    """Liste les algorithmes disponibles."""
    return {"algorithms": list(ALGORITHMS.keys())}
