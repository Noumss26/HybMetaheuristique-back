from pydantic import BaseModel, Field
from typing import List, Optional


# ─────────────────────────────────────────────
# City
# ─────────────────────────────────────────────
class City(BaseModel):
    """Représente une ville avec ses coordonnées."""
    name: str = Field(..., example="Paris")
    x: float = Field(..., example=340.0)
    y: float = Field(..., example=120.0)


# ─────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────
class OptimizeRequest(BaseModel):
    """Corps de la requête POST /optimize."""

    cities: List[City] = Field(
        ..., 
        min_length=2,
        description="Liste des villes à optimiser"
    )

    algorithm: str = Field(
        default="antcolony",
        description="Algorithme à utiliser : antcolony | genetic | hybrid | local_search"
    )

    start_city: Optional[str] = Field(
        default=None,
        description="Ville de départ (optionnelle). Si None → départ automatique (première ville du chemin optimal)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "algorithm": "antcolony",
                "start_city": "Paris",
                "cities": [
                    {"name": "Paris", "x": 340, "y": 120},
                    {"name": "Lyon", "x": 370, "y": 260},
                    {"name": "Marseille", "x": 390, "y": 380},
                    {"name": "Bordeaux", "x": 180, "y": 300},
                    {"name": "Lille", "x": 310, "y": 60}
                ]
            }
        }
    }


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────
class OptimizeResponse(BaseModel):
    """Réponse renvoyée après optimisation."""

    optimal_path: List[str]
    total_distance: float

    random_path: List[str]
    random_distance: float
    improvement_percent: float

    algorithm_used: str
    execution_time_ms: float

    # Ville de départ utilisée (très utile pour le frontend)
    start_city: Optional[str] = Field(
        default=None,
        description="Ville de départ appliquée (None = automatique)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "optimal_path": ["Paris", "Lille", "Lyon", "Marseille", "Bordeaux"],
                "total_distance": 1248.65,
                "random_path": ["Lyon", "Bordeaux", "Marseille", "Lille", "Paris"],
                "random_distance": 1890.34,
                "improvement_percent": 33.94,
                "algorithm_used": "antcolony",
                "execution_time_ms": 45.67,
                "start_city": "Paris"
            }
        }
    }