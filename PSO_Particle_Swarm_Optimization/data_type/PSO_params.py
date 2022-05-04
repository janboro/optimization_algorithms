from dataclasses import dataclass, field
from data_type.constriction_coefficients import ConstrictionCoefficients

constriction_coefficients = ConstrictionCoefficients()


@dataclass
class PSOParams:
    iterations: int = 100
    swarm_size: int = 50
    inertia: float = constriction_coefficients.xi
    cognitive_acceleration: float = constriction_coefficients.xi * constriction_coefficients.phi1
    social_acceleration: float = constriction_coefficients.xi * constriction_coefficients.phi2
    inertia_dampening: float = field(init=False)

    def __post_init__(self):
        self.inertia_dampening = 1 - (self.inertia / self.iterations)
