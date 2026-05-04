"""
Recherche locale 2-opt — version professionnelle.

Trois optimisations majeures par rapport à l'implémentation naïve :

1. **Delta evaluation O(1)** (Croes 1958, Lin 1965)
   Le 2-opt naïf recalcule la distance totale du tour à chaque candidat (O(n)).
   La version professionnelle ne calcule que la VARIATION de distance :

       Δ = d(t_i, t_k) + d(t_{i+1}, t_{k+1})
         − d(t_i, t_{i+1}) − d(t_k, t_{k+1})

   On ne fait l'inversion physique du segment QUE si Δ < 0 (gain).
   Complexité : O(n³) → O(n²) par passe complète.

2. **Don't-Look Bits** (Bentley 1992, "Fast Algorithms for the TSP")
   Chaque ville porte un bit indiquant si le dernier essai de move
   améliorant a échoué. Les villes « endormies » sont skipées tant
   qu'aucun de leurs voisins du tour n'a bougé. Empiriquement, réduit
   le travail par un facteur 4 à 10.

3. **Graph-aware delta**
   Quand un graphe partiel est fourni, on vérifie la validité des deux
   NOUVELLES arêtes en O(1) avant tout calcul. Move rejeté immédiatement
   si l'une des deux n'existe pas.

Stratégie : first-improvement (premier move améliorant accepté). En
pratique, plus rapide que best-improvement et qualité finale similaire
sur le TSP euclidien.
"""

from typing import List
from .utils import Tour, DistMatrix, Graph, tour_distance


def two_opt(
    tour: Tour,
    dist_matrix: DistMatrix,
    graph: Graph = None,
    max_iterations: int = 1000,
) -> Tour:
    """
    2-opt avec delta O(1) et Don't-Look Bits.

    Args:
        tour:         tour initial (permutation des indices de villes)
        dist_matrix:  matrice de distances précalculée (utils.build_distance_matrix)
        graph:        adjacence partielle, ou None pour graphe complet
        max_iterations: borne supérieure sur le nombre de passes complètes

    Returns:
        tour 2-optimal (aucun move 2-opt améliorant restant)
    """
    n = len(tour)
    if n < 4:
        return tour[:]  # 2-opt n'a aucun sens sur < 4 villes

    best = tour[:]

    # Don't-Look Bit indexé par VILLE (et non par position du tour) :
    # quand on swap, ce sont les 4 villes des arêtes échangées qu'on réveille.
    dont_look = [False] * n

    iteration = 0
    improved = True

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # On itère sur les positions i ; pour chaque i, on cherche un k
        # tel que le swap 2-opt (i, k) soit améliorant.
        for i in range(n - 1):
            city_i = best[i]
            if dont_look[city_i]:
                continue  # ville endormie, on skip

            found_move = False
            a = best[i]
            b = best[(i + 1) % n]
            d_ab = dist_matrix[a][b]  # arête courante 1, hoistée

            for k in range(i + 2, n):
                # Cas dégénéré : (i=0, k=n−1) inverse tout le tour, équivalent
                # en TSP cyclique symétrique → exclu pour éviter le no-op.
                if i == 0 and k == n - 1:
                    continue

                c = best[k]
                d = best[(k + 1) % n]

                # ── Filtrage graphe partiel : O(1) ───────────────────────
                # On rejette le move si l'une des deux nouvelles arêtes
                # (a,c) ou (b,d) n'est pas autorisée par le graphe.
                if graph is not None:
                    if c not in graph.get(a, set()):
                        continue
                    if d not in graph.get(b, set()):
                        continue

                # ── Delta evaluation O(1) ────────────────────────────────
                # Δ = (nouvelles arêtes) − (anciennes arêtes)
                # Move améliorant ssi Δ < 0.
                delta = (
                    dist_matrix[a][c] + dist_matrix[b][d]
                    - d_ab - dist_matrix[c][d]
                )

                if delta < -1e-12:  # tolérance numérique
                    # Inversion physique du segment [i+1, k] — l'unique
                    # opération O(n) du 2-opt, faite UNIQUEMENT à l'acceptation.
                    best[i + 1 : k + 1] = best[i + 1 : k + 1][::-1]

                    # Réveil des 4 villes impactées par le move
                    dont_look[a] = False
                    dont_look[b] = False
                    dont_look[c] = False
                    dont_look[d] = False

                    improved = True
                    found_move = True
                    break  # first-improvement : on relance la boucle externe

            if not found_move:
                # Aucun voisin de city_i n'a donné de gain → on l'endort.
                dont_look[city_i] = True

    return best


# ─────────────────────────────────────────────────────────────────────────────
# API legacy : conservée pour compat éventuelle, mais déconseillée
# ─────────────────────────────────────────────────────────────────────────────
def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
    """
    Effectue le swap 2-opt en construisant un NOUVEAU tour (allocation).
    Utilisée seulement pour démonstration / tests unitaires.
    Le 2-opt principal fait l'inversion in-place (plus rapide).
    """
    return tour[: i + 1] + tour[i + 1 : k + 1][::-1] + tour[k + 1 :]
