import random
from typing import List, Tuple, Optional

from .utils import Tour, tour_distance, Graph, get_neighbors


def fitness(tour: Tour, coords: List[Tuple[float, float]], graph: Graph = None) -> float:
    dist = tour_distance(tour, coords, graph)
    return 1.0 / dist if dist not in (0, float("inf")) else 0.0


def tournament_selection(
    population: List[Tour],
    coords: List[Tuple[float, float]],
    graph: Graph = None,
    k: int = 3,
) -> Tour:
    candidates = random.sample(population, k)
    return min(candidates, key=lambda t: tour_distance(t, coords, graph))


def order_crossover(parent1: Tour, parent2: Tour) -> Tour:
    """OX crossover — produit un tour valide en termes d'indices,
    mais pas forcément valide pour le graphe (filtré après)."""
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[start:end] = parent1[start:end]
    segment_set = set(parent1[start:end])
    fill_genes = [g for g in parent2 if g not in segment_set]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_genes[idx]
            idx += 1
    return child


def swap_mutation(tour: Tour, mutation_rate: float = 0.02) -> Tour:
    if random.random() < mutation_rate:
        tour = tour[:]
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def _random_valid_tour(n: int, graph: Graph) -> Optional[Tour]:
    """
    Génère un tour valide par DFS/backtracking sur le graphe.
    Retourne None si aucun tour hamiltonien n'existe depuis ce départ.
    Limité à de petits graphes ; pour de grands graphes, utilisez l'ACO.
    """
    if graph is None:
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    start = random.randint(0, n - 1)
    visited = [False] * n
    visited[start] = True
    path = [start]

    def backtrack() -> bool:
        if len(path) == n:
            # Vérifier la boucle de retour
            return start in graph.get(path[-1], set())
        current = path[-1]
        neighbors = [nb for nb in graph.get(current, set()) if not visited[nb]]
        random.shuffle(neighbors)
        for nb in neighbors:
            visited[nb] = True
            path.append(nb)
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        return False

    return path if backtrack() else None


def random_valid_population(size: int, n: int, graph: Graph) -> List[Tour]:
    """
    Construit une population de tours valides.
    Si le graphe est trop contraint, avertit et retourne moins d'individus.
    """
    population = []
    attempts = 0
    max_attempts = size * 20

    while len(population) < size and attempts < max_attempts:
        tour = _random_valid_tour(n, graph)
        if tour is not None:
            population.append(tour)
        attempts += 1

    return population


def run_genetic_algorithm(
    coords: List[Tuple[float, float]],
    graph: Graph = None,
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.02,
    elitism_count: int = 2,
    initial_population: List[Tour] = None,
) -> Tour:
    n = len(coords)

    if initial_population is not None:
        population = initial_population
    else:
        population = random_valid_population(population_size, n, graph)

    if not population:
        return []  # graphe sans solution hamiltonienne

    for _ in range(generations):
        population.sort(key=lambda t: tour_distance(t, coords, graph))

        next_gen: List[Tour] = population[:elitism_count]

        while len(next_gen) < population_size:
            parent1 = tournament_selection(population, coords, graph)
            parent2 = tournament_selection(population, coords, graph)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)

            # N'accepter l'enfant que s'il est valide pour le graphe
            if tour_distance(child, coords, graph) < float("inf"):
                next_gen.append(child)

        population = next_gen

    return min(population, key=lambda t: tour_distance(t, coords, graph))