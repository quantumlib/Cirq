# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Attempt to tabulate single qubit gates required to generate a target 2Q gate
with a product A k A.

Attention: this module is now deprecated! Use the classes from cirq-core instead."""
from typing import Tuple, NamedTuple

from dataclasses import dataclass
import numpy as np

import cirq
from cirq import TwoQubitGateTabulation, TwoQubitGateTabulationResult, two_qubit_gate_product_tabulation
from cirq._compat import deprecated, deprecated_class

_SingleQubitGatePair = Tuple[np.ndarray, np.ndarray]


class TwoQubitGateCompilation(TwoQubitGateTabulationResult):
    r"""Represents a compilation of a target 2-qubit with respect to a base
    gate.

    This object encodes the relationship between 4x4 unitary operators

    U_target ~ k_N · U_base · k_{N-1} · ... · k_1 · U_base · k_0

    where U_target, U_base are 2-local and k_j are 1-local.

    Attributes:
        base_gate: 4x4 unitary denoting U_base above.
        target_gate: 4x4 unitary denoting U_target above.
        local_unitaries: Sequence of 2-tuples
            $(k_{00}, k_{01}), (k_{10}, k_{11}) \ldots$ where
            $k_j = k_{j0} \otimes k_{j1}$ in the product above.
            Each $k_{j0}, k_{j1}$ is a 2x2 unitary.
        actual_gate: 4x4 unitary denoting the right hand side above, ideally
            equal to U_target.
        success: Whether actual_gate is expected to be close to U_target.
    """
    base_gate_unitary: np.ndarray
    target_gate: np.ndarray
    local_unitaries: Tuple[_SingleQubitGatePair, ...]
    actual_gate: np.ndarray
    success: bool


@dataclass
@deprecated_class(
    deadline='v0.16', fix='Use cirq.TwoQubitGateTabulation instead.', name='GateTabulation'
)
class GateTabulation(TwoQubitGateTabulation):
    pass


@deprecated(
    deadline='v0.16',
    fix='Use cirq.two_qubit_gate_product_tabulation instead.',
    name='gate_product_tabulation',
)
def gate_product_tabulation(
    base_gate: np.ndarray,
    max_infidelity: float,
    *,
    sample_scaling: int = 50,
    allow_missed_points: bool = True,
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> GateTabulation:
    r"""Generate a GateTabulation for a base two qubit unitary.

    Args:
        base_gate: The base gate of the tabulation.
        max_infidelity: Sets the desired density of tabulated product unitaries.
            The typical nearest neighbor Euclidean spacing (of the KAK vectors)
            will be on the order of $\sqrt{max\_infidelity}$. Thus the number of
            tabulated points will scale as $max\_infidelity^{-3/2}$.
        sample_scaling: Relative number of random gate products to use in the
            tabulation. The total number of random local unitaries scales as
            ~ $max\_infidelity^{-3/2} * sample\_scaling$. Must be positive.
        random_state: Random state or random state seed.
        allow_missed_points: If True, the tabulation is allowed to conclude
            even if not all points in the Weyl chamber are expected to be
            compilable using 2 or 3 base gates. Otherwise an error is raised
            in this case.

    Returns:
        A GateTabulation object used to compile new two-qubit gates from
        products of the base gate with 1-local unitaries.

    Raises:
        ValueError: If `allow_missed_points` is False and not all points
            in the Weyl chamber were compilable using 2 or 3 base gates.
    """
    result = two_qubit_gate_product_tabulation(
        base_gate,
        max_infidelity,
        sample_scaling=sample_scaling,
        allow_missed_points=allow_missed_points,
        random_state=random_state,
    )
    return GateTabulation(
        result.base_gate,
        result.kak_vecs,
        result.single_qubit_gates,
        result.max_expected_infidelity,
        result.summary,
        tuple(result.missed_points),
    )
