from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConstrictionCoefficients:
    kappa: float = 1
    phi1: float = 2.05
    phi2: float = 2.05
    xi: float = field(init=False)

    def __post_init__(self):
        self._validate_kappa()
        self._validate_phi()

        phi = self.phi1 + self.phi2
        self.xi = 2 * self.kappa / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))

    def _validate_kappa(self):
        if self.kappa < 0 or self.kappa > 1:
            raise ValueError("Kappa should be between 0 <= kappa <= 1")

    def _validate_phi(self):
        if self.phi1 + self.phi2 < 4:
            raise ValueError("The sum of phi1 and phi2 should be greater than 4")
