"""
algorithms/hybrid.py
Pipeline d'hybridation à trois phases :

  Phase 1 — ACO  : exploration par phéromones → solution de qualité
  Phase 2 — GA   : évolution à partir de la solution ACO + population mixte
  Phase 3 — 2-opt: affinage local du meilleur résultat GA

Chaque algorithme compense les faiblesses des autres :
  - ACO    : bon pour converger vite, sensible aux minima locaux
  - GA     : diversité génétique, échappe aux minima locaux
  - 2-opt  : garantit l'absence de croisements dans la solution finale
"""

import logging
from typing import List, Tuple

from .antcolony import run_ant_colony
from .genetic import run_genetic_algorithm
from .local_search import two_opt
from .utils import Tour, tour_distance, random_population

logger = logging.getLogger("optimizer.hybrid")


# ---------------------------------------------------------------------------
# Initialisation de population mixte (ACO + aléatoire)
# ---------------------------------------------------------------------------

def _build_seeded_population(
    aco_tour: Tour,
    population_size: int,
    n_cities: int,
) -> List[Tour]:
    """
    Crée une population initiale pour le GA en mélangeant :
    - la solution ACO (injectée plusieurs fois pour peser dans la sélection)
    - des tours aléatoires (diversité)
    """
    n_aco_seeds = max(1, population_size // 5)
    aco_seeds = [aco_tour[:] for _ in range(n_aco_seeds)]
    random_tours = random_population(population_size - n_aco_seeds, n_cities)
    return aco_seeds + random_tours


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_hybrid_optimization(
    coords: List[Tuple[float, float]],
    # --- paramètres ACO ---
    aco_ants: int = 30,
    aco_iterations: int = 100,
    aco_alpha: float = 1.0,
    aco_beta: float = 2.0,
    aco_evaporation: float = 0.5,
    aco_q: float = 100.0,
    # --- paramètres GA ---
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    elitism_count: int = 2,
    # --- paramètres 2-opt ---
    max_2opt_iterations: int = 1000,
) -> Tuple[Tour, float]:
    """
    Optimise un trajet TSP via hybridation ACO + GA + 2-opt.
    Retourne (best_tour, total_distance).
    """
    n = len(coords)

    # ── Phase 1 : ACO ────────────────────────────────────────────────────
    logger.info("🐜 Phase 1 : ACO démarré (%d fourmis, %d itérations)", aco_ants, aco_iterations)

    aco_tour = run_ant_colony(
        coords,
        n_ants=aco_ants,
        iterations=aco_iterations,
        alpha=aco_alpha,
        beta=aco_beta,
        evaporation=aco_evaporation,
        q=aco_q,
    )

    aco_distance = tour_distance(aco_tour, coords)
    logger.info("✅ ACO terminé — distance : %.4f", aco_distance)

    # ── Phase 2 : GA amorcé par la solution ACO ──────────────────────────
    logger.info("🧬 Phase 2 : GA démarré (%d générations, pop=%d)", generations, population_size)

    seeded_population = _build_seeded_population(aco_tour, population_size, n)

    ga_tour = run_genetic_algorithm(
        coords,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
        initial_population=seeded_population,
    )

    ga_distance = tour_distance(ga_tour, coords)
    logger.info("✅ GA terminé — distance : %.4f (ACO était %.4f)", ga_distance, aco_distance)

    # Garder le meilleur entre ACO pur et GA
    if ga_distance <= aco_distance:
        best_after_ga = ga_tour
        logger.info("🏆 GA meilleur que ACO")
    else:
        best_after_ga = aco_tour
        logger.warning("⚠️ ACO meilleur que GA (GA n'a pas amélioré) — vérifiez les paramètres")

    # ── Phase 3 : 2-opt ──────────────────────────────────────────────────
    logger.info("🔧 Phase 3 : 2-opt démarré (max %d itérations)", max_2opt_iterations)

    optimized_tour = two_opt(
        best_after_ga,
        coords,
        max_2opt_iterations,
    )

    total_distance = tour_distance(optimized_tour, coords)
    logger.info("✅ 2-opt terminé — distance finale : %.4f", total_distance)

    return optimized_tour, total_distance