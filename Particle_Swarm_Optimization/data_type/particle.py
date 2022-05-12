from pydantic import BaseModel
import numpy as np
from data_type.best_solution import BestSolution


class Particle(BaseModel):
    position: np.ndarray
    velocity: np.ndarray
    cost: float
    personal_best: BestSolution

    class Config:
        arbitrary_types_allowed = True
