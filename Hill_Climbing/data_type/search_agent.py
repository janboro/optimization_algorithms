from dataclasses import dataclass
from numpy.typing import ArrayLike


@dataclass
class SearchAgent:
    position: ArrayLike
    cost: float
