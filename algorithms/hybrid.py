import logging
from typing import List, Tuple

from .antcolony import run_ant_colony
from .genetic import run_genetic_algorithm, random_valid_population
from .local_search import two_opt
from .utils import Tour, tour_distance, Graph

logger = logging.getLogger("optimizer.hybrid")


def _build_seeded_population(
    aco_tour: Tour,
    population_size: int,
    n_cities: int,
    graph: Graph = None,
) -> List[Tour]:
    n_aco_seeds = max(1, population_size // 5)
    aco_seeds = [aco_tour[:] for _ in range(n_aco_seeds)]
    random_tours = random_valid_population(population_size - n_aco_seeds, n_cities, graph)
    return aco_seeds + random_tours


def run_hybrid_optimization(
    coords: List[Tuple[float, float]],
    graph: Graph = None,
    aco_ants: int = 30,
    aco_iterations: int = 100,
    aco_alpha: float = 1.0,
    aco_beta: float = 2.0,
    aco_evaporation: float = 0.5,
    aco_q: float = 100.0,
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    elitism_count: int = 2,
    max_2opt_iterations: int = 1000,
) -> Tuple[Tour, float]:
    n = len(coords)

    logger.info("🐜 Phase 1 : ACO (%d fourmis, %d itérations)", aco_ants, aco_iterations)
    aco_tour = run_ant_colony(coords, graph=graph, n_ants=aco_ants, iterations=aco_iterations,
                              alpha=aco_alpha, beta=aco_beta, evaporation=aco_evaporation, q=aco_q)

    if not aco_tour:
        raise ValueError("ACO n'a trouvé aucun tour valide — vérifiez la connexité du graphe.")

    aco_distance = tour_distance(aco_tour, coords, graph)
    logger.info("✅ ACO terminé — distance : %.4f", aco_distance)

    logger.info("🧬 Phase 2 : GA (%d générations)", generations)
    seeded_population = _build_seeded_population(aco_tour, population_size, n, graph)
    ga_tour = run_genetic_algorithm(coords, graph=graph, population_size=population_size,
                                    generations=generations, mutation_rate=mutation_rate,
                                    elitism_count=elitism_count, initial_population=seeded_population)

    ga_distance = tour_distance(ga_tour, coords, graph) if ga_tour else float("inf")
    logger.info("✅ GA terminé — distance : %.4f", ga_distance)

    best_after_ga = ga_tour if ga_distance <= aco_distance else aco_tour

    logger.info("🔧 Phase 3 : 2-opt")
    optimized_tour = two_opt(best_after_ga, coords, max_2opt_iterations, graph)
    total_distance = tour_distance(optimized_tour, coords, graph)
    logger.info("✅ 2-opt terminé — distance finale : %.4f", total_distance)

    return optimized_tour, total_distance