# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Attempt to tabulate single qubit gates required to generate a target 2Q gate
with a product A k A.

Attention: this module is now deprecated! Use the classes from cirq-core instead."""
from typing import cast
import numpy as np

import cirq
from cirq import (
    TwoQubitGateTabulation,
    TwoQubitGateTabulationResult,
    two_qubit_gate_product_tabulation,
)
from cirq._compat import deprecated, deprecated_class


@deprecated_class(
    deadline='v0.16',
    fix='Use cirq.TwoQubitGateTabulationResult instead.',
    name='TwoQubitGateCompilation',
)
class TwoQubitGateCompilation(TwoQubitGateTabulationResult):
    pass


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
    return cast(
        GateTabulation,
        two_qubit_gate_product_tabulation(
            base_gate,
            max_infidelity,
            sample_scaling=sample_scaling,
            allow_missed_points=allow_missed_points,
            random_state=random_state,
        ),
    )
