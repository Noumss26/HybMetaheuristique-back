from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class City(BaseModel):
    name: str = Field(..., example="Paris")
    x: float = Field(..., example=340.0)
    y: float = Field(..., example=120.0)


class Edge(BaseModel):
    """
    Connexion bidirectionnelle entre deux villes.
    city_a <-> city_b (les deux sens sont toujours valides).
    """
    city_a: str = Field(..., example="Paris")
    city_b: str = Field(..., example="Bordeaux")


class OptimizeRequest(BaseModel):
    cities: List[City] = Field(..., min_length=2)
    edges: Optional[List[Edge]] = Field(
        default=None,
        description=(
            "Connexions autorisées entre villes (bidirectionnelles). "
            "Si None → graphe complet (comportement actuel)."
        )
    )
    algorithm: str = Field(default="antcolony")
    start_city: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "algorithm": "antcolony",
                "start_city": "Vienne",
                "cities": [
                    {"name": "Vienne",    "x": 580, "y": 180},
                    {"name": "Paris",     "x": 340, "y": 120},
                    {"name": "Bordeaux",  "x": 180, "y": 300},
                    {"name": "Lyon",      "x": 370, "y": 260},
                    {"name": "Marseille", "x": 390, "y": 380},
                ],
                "edges": [
                    {"city_a": "Vienne",   "city_b": "Paris"},
                    {"city_a": "Paris",    "city_b": "Bordeaux"},
                    {"city_a": "Paris",    "city_b": "Lyon"},
                    {"city_a": "Lyon",     "city_b": "Marseille"},
                    {"city_a": "Bordeaux", "city_b": "Marseille"},
                ]
            }
        }
    }


class AlgoBreakdown(BaseModel):
    """Résultat d'un algorithme individuel pour la comparaison frontend."""
    name: str
    distance: float
    time_ms: float


class OptimizeResponse(BaseModel):
    optimal_path: List[str]
    total_distance: float
    random_path: List[str]
    random_distance: float
    improvement_percent: float
    algorithm_used: str
    execution_time_ms: float
    start_city: Optional[str] = None
    breakdown: List[AlgoBreakdown] = Field(default_factory=list)