from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Bounds:
    upper: float
    lower: float
    constrain_velocity: bool = True
    velocity_constraint_multiplier: float = 0.2
    max_velocity: float = field(init=False)
    min_velocity: float = field(init=False)

    def __post_init__(self):
        velocity_limit = self.velocity_constraint_multiplier * (self.upper - self.lower)
        self.max_velocity = velocity_limit
        self.min_velocity = -velocity_limit


@dataclass
class Problem:
    cost_function: Callable
    variables: int
    bounds: Bounds
