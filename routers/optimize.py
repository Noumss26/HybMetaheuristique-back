import time
import math
import logging
from fastapi import APIRouter, HTTPException
from models.city import OptimizeRequest, OptimizeResponse

from algorithms.antcolony import run_ant_colony
from algorithms.genetic import run_genetic_algorithm, random_valid_population
from algorithms.hybrid import run_hybrid_optimization
from algorithms.local_search import two_opt
from algorithms.utils import build_graph, tour_distance

logger = logging.getLogger("optimizer")
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/optimize", tags=["Optimisation"])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _total_distance(path, coords, graph=None):
    if not path or len(path) < 2:
        return 0.0
    return round(tour_distance(path, coords, graph), 4)


def _rotate_to_start(tour, start_index):
    if start_index is None or start_index not in tour:
        return tour
    idx = tour.index(start_index)
    return tour[idx:] + tour[:idx]


# ─────────────────────────────────────────────────────────────
# Algorithmes wrappers
# ─────────────────────────────────────────────────────────────
def _run_antcolony(coords, graph):
    tour = run_ant_colony(coords, graph=graph)
    return tour, _total_distance(tour, coords, graph)


def _run_genetic(coords, graph):
    tour = run_genetic_algorithm(coords, graph=graph)
    return tour, _total_distance(tour, coords, graph)


def _run_hybrid(coords, graph):
    return run_hybrid_optimization(coords, graph=graph)


def _run_local_search(coords, graph):
    n = len(coords)
    candidates = random_valid_population(1, n, graph)
    initial = candidates[0] if candidates else list(range(n))
    tour = two_opt(initial, coords, graph=graph)
    return tour, _total_distance(tour, coords, graph)


ALGORITHMS = {
    "antcolony": _run_antcolony,
    "genetic": _run_genetic,
    "hybrid": _run_hybrid,
    "local_search": _run_local_search,
}


# ─────────────────────────────────────────────────────────────
# Route principale
# ─────────────────────────────────────────────────────────────
@router.post("", response_model=OptimizeResponse)
async def optimize(payload: OptimizeRequest) -> OptimizeResponse:

    coords = [(c.x, c.y) for c in payload.cities]
    names = [c.name for c in payload.cities]
    n = len(coords)

    if n == 0:
        raise HTTPException(status_code=400, detail="Aucune ville fournie.")

    # ── Graphe ────────────────────────────────────────────────
    graph = None

    if payload.edges:
        edge_list = []
        for edge in payload.edges:
            if edge.city_a not in names or edge.city_b not in names:
                raise HTTPException(status_code=400, detail="Arête invalide")

            a_idx = names.index(edge.city_a)
            b_idx = names.index(edge.city_b)
            edge_list.append((a_idx, b_idx))

        graph = build_graph(edge_list, n)
        logger.info("🗺️ Graphe partiel : %d arêtes", len(edge_list))
    else:
        logger.info("🗺️ Graphe complet")

    # ── Start city ────────────────────────────────────────────
    start_index = None
    if payload.start_city:
        if payload.start_city not in names:
            raise HTTPException(status_code=400, detail="start_city invalide")
        start_index = names.index(payload.start_city)

    # ─────────────────────────────────────────────────────────
    # 🔥 TEST DE TOUS LES ALGORITHMES
    # ─────────────────────────────────────────────────────────
    results = []
    errors = {}

    t_start = time.perf_counter()

    for name, algo in ALGORITHMS.items():
        try:
            logger.info(f"⚙️ Test : {name}")

            tour, dist = algo(coords, graph)

            if not tour or len(tour) != n:
                raise ValueError("Tour invalide ou incomplet")

            results.append({
                "name": name,
                "tour": tour,
                "distance": dist
            })

        except Exception as e:
            logger.warning(f"❌ {name} échoué : {e}")
            errors[name] = str(e)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # ── Aucun algo valide ─────────────────────────────────────
    if not results:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Tous les algorithmes ont échoué",
                "errors": errors
            }
        )

    # ── Meilleur résultat ─────────────────────────────────────
    best = min(results, key=lambda x: x["distance"])

    tour_indices = best["tour"]
    optimal_distance = best["distance"]
    algorithm_used = best["name"]

    # ── Rotation ─────────────────────────────────────────────
    tour_indices = _rotate_to_start(tour_indices, start_index)

    optimal_path = [names[i] for i in tour_indices]

    # ── Random solution ──────────────────────────────────────
    rand_candidates = random_valid_population(1, n, graph)
    random_indices = rand_candidates[0] if rand_candidates else list(range(n))
    random_indices = _rotate_to_start(random_indices, start_index)

    random_path = [names[i] for i in random_indices]
    random_distance = _total_distance(random_indices, coords, graph)

    improvement = (
        round(((random_distance - optimal_distance) / random_distance) * 100, 2)
        if random_distance > 0 else 0.0
    )

    # ── Réponse ──────────────────────────────────────────────
    return OptimizeResponse(
        optimal_path=optimal_path,
        total_distance=optimal_distance,
        algorithm_used=algorithm_used,
        execution_time_ms=elapsed_ms,
        random_path=random_path,
        random_distance=random_distance,
        improvement_percent=improvement,
        start_city=payload.start_city,
        errors=errors  # 🔥 important pour frontend
    )


# ─────────────────────────────────────────────────────────────
@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": list(ALGORITHMS.keys())}