"""
Hybridation séquentielle ACO → GA → 2-opt — version professionnelle.

Pipeline en trois phases, chacune exploitant la sortie de la précédente :

  Phase 1 (DIVERSIFICATION+INTENSIFICATION) — ACO
    Les fourmis explorent l'espace via phéromones et heuristique 1/d.
    Sortie : un tour de bonne qualité globale.

  Phase 2 (RECOMBINAISON) — GA seedé
    La population initiale est composée :
      • ~20 % de copies du tour ACO (« seeding »)
      • ~80 % de tours aléatoires valides
    Le GA recombine ces blocs pour potentiellement échapper aux optima
    de l'ACO via crossover OX et mutation par inversion.

  Phase 3 (INTENSIFICATION FINALE) — 2-opt
    Recherche locale déterministe avec delta evaluation O(1) et
    Don't-Look Bits : raffine le meilleur tour ACO/GA jusqu'à
    optimum 2-local.

Cette stratégie correspond au schéma classique « metaheuristic + local
search » (memetic algorithm). La matrice de distances est précalculée
UNE FOIS et partagée par les trois phases pour des performances
maximales.
"""

import logging
from typing import List, Tuple

from .antcolony import run_ant_colony
from .genetic import run_genetic_algorithm, random_valid_population
from .local_search import two_opt
from .utils import Tour, DistMatrix, Graph, tour_distance

logger = logging.getLogger("optimizer.hybrid")


# ─────────────────────────────────────────────────────────────────────────────
# Construction de la population « seedée » pour le GA
# ─────────────────────────────────────────────────────────────────────────────
def _build_seeded_population(
    aco_tour: Tour,
    population_size: int,
    n_cities: int,
    graph: Graph,
    seed_ratio: float = 0.2,
) -> List[Tour]:
    """
    Population initiale du GA :
      • `seed_ratio` × population_size copies du tour ACO
      • le reste : tours aléatoires valides (graphe-aware)

    Le seeding accélère la convergence du GA en injectant directement
    une bonne solution initiale, mais on garde une majorité de hasard
    pour préserver la diversité génétique (sinon convergence prématurée).
    """
    n_seeds = max(1, int(population_size * seed_ratio))
    seeds = [aco_tour[:] for _ in range(n_seeds)]
    randoms = random_valid_population(population_size - n_seeds, n_cities, graph)
    return seeds + randoms


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline hybride
# ─────────────────────────────────────────────────────────────────────────────
def run_hybrid_optimization(
    dist_matrix: DistMatrix,
    n_cities: int,
    graph: Graph = None,
    # ── Hyperparamètres ACO ──────────────────────────────────────────
    aco_ants: int = 30,
    aco_iterations: int = 100,
    aco_alpha: float = 1.0,
    aco_beta: float = 2.0,
    aco_evaporation: float = 0.5,
    aco_q: float = 100.0,
    # ── Hyperparamètres GA ───────────────────────────────────────────
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    elitism_count: int = 2,
    # ── Hyperparamètres 2-opt ────────────────────────────────────────
    max_2opt_iterations: int = 1000,
) -> Tuple[Tour, float]:
    """
    Exécute le pipeline hybride et retourne (tour_optimisé, distance).

    Robustesse : si une phase échoue (ex. ACO ne trouve aucun tour valide
    sur graphe trop contraint), on lève une ValueError explicite plutôt
    que de retourner un tour aberrant.
    """
    # ── Phase 1 : Ant Colony Optimization ────────────────────────────
    logger.info("🐜 Phase 1 : ACO (%d fourmis × %d itérations)", aco_ants, aco_iterations)
    aco_tour = run_ant_colony(
        dist_matrix=dist_matrix,
        n_cities=n_cities,
        graph=graph,
        n_ants=aco_ants,
        iterations=aco_iterations,
        alpha=aco_alpha,
        beta=aco_beta,
        evaporation=aco_evaporation,
        q=aco_q,
    )

    if not aco_tour:
        raise ValueError(
            "ACO n'a trouvé aucun tour hamiltonien valide. "
            "Vérifiez la connexité du graphe."
        )

    aco_dist = tour_distance(aco_tour, dist_matrix, graph)
    logger.info("✅ ACO terminé — distance = %.4f", aco_dist)

    # ── Phase 2 : Genetic Algorithm seedé ────────────────────────────
    logger.info("🧬 Phase 2 : GA (%d gén. × %d ind.)", generations, population_size)
    seeded_pop = _build_seeded_population(aco_tour, population_size, n_cities, graph)
    ga_tour = run_genetic_algorithm(
        dist_matrix=dist_matrix,
        n_cities=n_cities,
        graph=graph,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
        initial_population=seeded_pop,
    )

    ga_dist = tour_distance(ga_tour, dist_matrix, graph) if ga_tour else float("inf")
    logger.info("✅ GA terminé — distance = %.4f", ga_dist)

    # On garde le meilleur des deux phases comme entrée du 2-opt :
    # le GA peut occasionnellement régresser malgré l'élitisme si la
    # population se dégrade par mutation excessive sur graphe contraint.
    best_so_far = ga_tour if ga_dist <= aco_dist else aco_tour

    # ── Phase 3 : 2-opt avec delta O(1) ──────────────────────────────
    logger.info("🔧 Phase 3 : 2-opt (delta O(1) + DLB)")
    final_tour = two_opt(
        best_so_far,
        dist_matrix=dist_matrix,
        graph=graph,
        max_iterations=max_2opt_iterations,
    )
    final_dist = tour_distance(final_tour, dist_matrix, graph)
    logger.info("✅ 2-opt terminé — distance finale = %.4f", final_dist)

    return final_tour, final_dist
