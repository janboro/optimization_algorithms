import numpy as np
from pydantic import BaseModel
from typing import Optional


class BestSolution(BaseModel):
    position: Optional[np.ndarray]
    cost: float

    class Config:
        arbitrary_types_allowed = True
