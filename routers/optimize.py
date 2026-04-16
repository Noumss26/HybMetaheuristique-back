# routers/optimize.py

import random
import time
import math
import logging
from fastapi import APIRouter, HTTPException
from models.city import OptimizeRequest, OptimizeResponse

from algorithms.antcolony import run_ant_colony
from algorithms.genetic import run_genetic_algorithm
from algorithms.hybrid import run_hybrid_optimization
from algorithms.local_search import two_opt

# ── Logger config ─────────────────────────────────────────

logger = logging.getLogger("optimizer")
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/optimize", tags=["Optimisation"])


# ── Helpers ──────────────────────────────────────────────

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _total_distance(path: list[int], coords: list[tuple[float, float]]) -> float:
    if not path or len(path) < 2:
        logger.warning("⚠️ Distance calculée sur path vide ou trop court")
        return 0.0

    total = 0.0
    n = len(path)

    for i in range(n):
        a = coords[path[i]]
        b = coords[path[(i + 1) % n]]
        total += _dist(a, b)

    return round(total, 4)


def _generate_random_solution(n: int) -> list[int]:
    indices = list(range(n))
    random.shuffle(indices)
    return indices


def _rotate_to_start(tour: list[int], start_index: int) -> list[int]:
    if start_index not in tour:
        return tour[:]
    idx = tour.index(start_index)
    return tour[idx:] + tour[:idx]


# ── Wrappers normalisés ──────────────────────────────────
# Chaque wrapper retourne toujours (tour_indices, distance)
# pour uniformiser le traitement dans l'endpoint.

def _run_antcolony(coords, **_) -> tuple[list[int], float]:
    tour = run_ant_colony(coords)
    return tour, _total_distance(tour, coords)


def _run_genetic(coords, **_) -> tuple[list[int], float]:
    tour = run_genetic_algorithm(coords)
    return tour, _total_distance(tour, coords)


def _run_hybrid(coords, **_) -> tuple[list[int], float]:
    # run_hybrid_optimization retourne déjà (tour, distance)
    return run_hybrid_optimization(coords)


def _run_local_search(coords, **_) -> tuple[list[int], float]:
    # 2-opt seul : on part d'une solution aléatoire comme base
    n = len(coords)
    initial = list(range(n))
    random.shuffle(initial)
    tour = two_opt(initial, coords)
    return tour, _total_distance(tour, coords)


ALGORITHMS = {
    "antcolony":    _run_antcolony,
    "genetic":      _run_genetic,
    "hybrid":       _run_hybrid,
    "local_search": _run_local_search,
}


# ── Endpoint principal ───────────────────────────────────

@router.post("", response_model=OptimizeResponse)
async def optimize(payload: OptimizeRequest) -> OptimizeResponse:
    algo_name = payload.algorithm.lower()

    logger.info("📥 Requête reçue — Algo: %s", algo_name)

    if algo_name not in ALGORITHMS:
        logger.error("❌ Algorithme inconnu: %s", algo_name)
        raise HTTPException(
            status_code=400,
            detail=f"Algorithme '{algo_name}' inconnu. Choix : {list(ALGORITHMS.keys())}",
        )

    coords = [(c.x, c.y) for c in payload.cities]
    names  = [c.name for c in payload.cities]
    n      = len(coords)

    if n == 0:
        raise HTTPException(status_code=400, detail="Aucune ville fournie.")

    # ── Ville de départ ───────────────────────────────────
    start_index: int | None = None
    if payload.start_city:
        if payload.start_city not in names:
            raise HTTPException(
                status_code=400,
                detail=f"start_city invalide : '{payload.start_city}'"
            )
        start_index = names.index(payload.start_city)
        logger.info("🚀 Ville de départ : %s (index %d)", payload.start_city, start_index)

    logger.info("📊 Nombre de villes : %d", n)

    # ── Exécution de l'algorithme ─────────────────────────
    algo_fn = ALGORITHMS[algo_name]
    t_start = time.perf_counter()

    try:
        tour_indices, optimal_distance = algo_fn(coords)
    except Exception as exc:
        logger.exception("💥 Erreur lors de l'exécution de '%s'", algo_name)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur algorithme '{algo_name}' : {exc}",
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    if not tour_indices or len(tour_indices) != n:
        raise HTTPException(
            status_code=500,
            detail="Tour invalide retourné par l'algorithme (longueur incorrecte).",
        )

    logger.info("✅ Tour optimal indices : %s", tour_indices)
    logger.info("📏 Distance optimale   : %.4f", optimal_distance)
    logger.info("⏱  Temps d'exécution   : %.2f ms", elapsed_ms)

    # ── Rotation vers la ville de départ ─────────────────
    if start_index is not None:
        tour_indices = _rotate_to_start(tour_indices, start_index)

    optimal_path = [names[i] for i in tour_indices]
    logger.info("🧭 Chemin optimal : %s", optimal_path)

    # ── Solution aléatoire (référence) ───────────────────
    random_indices = _generate_random_solution(n)
    if start_index is not None:
        random_indices = _rotate_to_start(random_indices, start_index)

    random_path     = [names[i] for i in random_indices]
    random_distance = _total_distance(random_indices, coords)

    logger.info("🎲 Random path     : %s", random_path)
    logger.info("📏 Random distance : %.4f", random_distance)

    # ── Amélioration relative ────────────────────────────
    if random_distance > 0:
        improvement = round(
            ((random_distance - optimal_distance) / random_distance) * 100, 2
        )
    else:
        improvement = 0.0

    logger.info("📈 Amélioration : %.2f%%", improvement)

    # ── Réponse ──────────────────────────────────────────
    return OptimizeResponse(
        optimal_path=optimal_path,
        total_distance=optimal_distance,
        algorithm_used=algo_name,
        execution_time_ms=elapsed_ms,
        random_path=random_path,
        random_distance=random_distance,
        improvement_percent=improvement,
    )


# ── Liste des algorithmes ─────────────────────────────────

@router.get("/algorithms")
async def list_algorithms():
    logger.info("📡 Liste des algorithmes demandée")
    return {"algorithms": list(ALGORITHMS.keys())}