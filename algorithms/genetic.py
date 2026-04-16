"""
algorithms/genetic.py
Algorithme génétique (GA) pour le problème du voyageur de commerce.
Fournit une bonne solution initiale que 2-opt affinera ensuite.
"""

import random
from typing import List, Tuple

from .utils import Tour, tour_distance, random_population


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def fitness(tour: Tour, coords: List[Tuple[float, float]]) -> float:
    """
    Calcule la fitness d'un individu.
    On inverse la distance : fitness élevée = trajet court.
    """
    dist = tour_distance(tour, coords)
    return 1.0 / dist if dist > 0 else float("inf")


# ---------------------------------------------------------------------------
# Sélection
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[Tour],
    coords: List[Tuple[float, float]],
    k: int = 3,
) -> Tour:
    """
    Sélection par tournoi : choisit k individus au hasard
    et retourne le meilleur (fitness max = distance min).

    Args:
        population: Population courante.
        coords:     Coordonnées des villes.
        k:          Taille du tournoi.

    Returns:
        Le meilleur tour parmi les k candidats.
    """
    candidates = random.sample(population, k)
    return min(candidates, key=lambda t: tour_distance(t, coords))


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def order_crossover(parent1: Tour, parent2: Tour) -> Tour:
    """
    OX (Order Crossover) : préserve l'ordre relatif des gènes.

    1. Copie un segment du parent1.
    2. Remplit le reste avec les villes du parent2 dans leur ordre.

    Args:
        parent1, parent2: Deux tours parents.

    Returns:
        Un tour enfant valide (pas de doublon).
    """
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))

    # Segment hérité du parent1
    child = [None] * n
    child[start:end] = parent1[start:end]

    # Compléter avec parent2 en préservant l'ordre
    segment_set = set(parent1[start:end])
    fill_genes = [gene for gene in parent2 if gene not in segment_set]

    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_genes[idx]
            idx += 1

    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def swap_mutation(tour: Tour, mutation_rate: float = 0.02) -> Tour:
    """
    Mutation par échange de deux villes.
    Appliquée avec une probabilité `mutation_rate`.

    Args:
        tour:          Tour à muter (non modifié sur place).
        mutation_rate: Probabilité de déclencher la mutation.

    Returns:
        Tour muté (copie si mutation, original sinon).
    """
    if random.random() < mutation_rate:
        tour = tour[:]  # copie pour ne pas modifier l'original
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


# ---------------------------------------------------------------------------
# Algorithme principal
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    coords: List[Tuple[float, float]],
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    elitism_count: int = 2,
    initial_population: List[Tour] = None,
) -> Tour:
    """
    Exécute l'algorithme génétique et retourne le meilleur tour trouvé.

    Stratégie :
      - Élitisme : les `elitism_count` meilleurs passent directement.
      - Le reste est généré par tournoi + OX crossover + mutation.

    Args:
        coords:             Liste de (x, y) pour chaque ville.
        population_size:    Nombre d'individus par génération.
        generations:        Nombre de générations.
        mutation_rate:      Probabilité de mutation par individu.
        elitism_count:      Nombre d'élites conservés à chaque génération.
        initial_population: Population de départ optionnelle (amorçage ACO).
                            Si None, une population aléatoire est générée.

    Returns:
        Meilleur tour trouvé (liste d'indices).
    """
    n = len(coords)
    population = (
        initial_population
        if initial_population is not None
        else random_population(population_size, n)
    )

    for _ in range(generations):
        # Trier par distance croissante (meilleur = plus court)
        population.sort(key=lambda t: tour_distance(t, coords))

        next_gen: List[Tour] = population[:elitism_count]  # élites

        while len(next_gen) < population_size:
            parent1 = tournament_selection(population, coords)
            parent2 = tournament_selection(population, coords)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            next_gen.append(child)

        population = next_gen

    # Retourner le meilleur individu
    return min(population, key=lambda t: tour_distance(t, coords))