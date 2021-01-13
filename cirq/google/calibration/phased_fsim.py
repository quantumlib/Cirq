from typing import Dict, Optional

import dataclasses


# TODO: Add JSON serialization support
@dataclasses.dataclass(frozen=True)
class PhasedFSimCharacterization:
    """Holder for the unitary angles of the cirq.PhasedFSimGate.

    This class stores five unitary parameters (θ, ζ, χ, γ and φ) that describe the
    cirq.PhasedFSimGate which is the most general particle conserving two-qubit gate. The unitary
    of the underlying gate is:

        [[1,                        0,                       0,                0],
         [0,    exp(-i(γ + ζ)) cos(θ), -i exp(-i(γ - χ)) sin(θ),               0],
         [0, -i exp(-i(γ + χ)) sin(θ),    exp(-i(γ - ζ)) cos(θ),               0],
         [0,                        0,                       0,  exp(-i(2γ + φ))]]

    The parameters θ, γ and φ are symmetric and parameters ζ and χ asymmetric under the qubits
    exchange.

    All the angles described by this class are optional and can be left unknown. This is relevant
    for characterization routines that characterize only subset of the gate parameters. All the
    angles are assumed to take a fixed numerical values which reflect the current state of the
    characterized gate.

    Attributes:
        theta: θ angle in radians or None when unknown.
        zeta: ζ angle in radians or None when unknown.
        chi: χ angle in radians or None when unknown.
        gamma: γ angle in radians or None when unknown.
        phi: φ angle in radians or None when unknown.
    """

    theta: Optional[float] = None
    zeta: Optional[float] = None
    chi: Optional[float] = None
    gamma: Optional[float] = None
    phi: Optional[float] = None

    def asdict(self) -> Dict[str, float]:
        """Converts parameters to a dictionary that maps angle names to values."""
        return dataclasses.asdict(self)

    def all_none(self) -> bool:
        """Returns True if all the angles are None"""
        return (
            self.theta is None
            and self.zeta is None
            and self.chi is None
            and self.gamma is None
            and self.phi is None
        )

    def any_none(self) -> bool:
        """Returns True if any the angle is None"""
        return (
            self.theta is None
            or self.zeta is None
            or self.chi is None
            or self.gamma is None
            or self.phi is None
        )

    def parameters_for_qubits_swapped(self) -> 'PhasedFSimCharacterization':
        """Parameters for the gate with qubits swapped between each other.

        The angles theta, gamma and phi are kept unchanged. The angles zeta and chi are negated for
        the gate with swapped qubits.

        Returns:
            New instance with angles adjusted for swapped qubits.
        """
        return PhasedFSimCharacterization(
            theta=self.theta,
            zeta=-self.zeta if self.zeta is not None else None,
            chi=-self.chi if self.chi is not None else None,
            gamma=self.gamma,
            phi=self.phi,
        )

    def merge_with(self, other: 'PhasedFSimCharacterization') -> 'PhasedFSimCharacterization':
        """Substitutes missing parameter with values from other.

        Args:
            other: Parameters to use for None values.

        Returns:
            New instance of PhasedFSimCharacterization with values from this instance if they are
            set or values from other when some parameter is None.
        """
        return PhasedFSimCharacterization(
            theta=other.theta if self.theta is None else self.theta,
            zeta=other.zeta if self.zeta is None else self.zeta,
            chi=other.chi if self.chi is None else self.chi,
            gamma=other.gamma if self.gamma is None else self.gamma,
            phi=other.phi if self.phi is None else self.phi,
        )

    def override_by(self, other: 'PhasedFSimCharacterization') -> 'PhasedFSimCharacterization':
        """Overrides other parameters that are not None.

        Args:
            other: Parameters to use for override.

        Returns:
            New instance of PhasedFSimCharacterization with values from other if set (values from
            other that are not None). Otherwise the current values are used.
        """
        return other.merge_with(self)
