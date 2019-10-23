"""Gates for Google's Sycamore architecture."""

from typing import TYPE_CHECKING

import numpy as np

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


class SycamoreGate(ops.FSimGate):
    """Sycamore two-qubit gate is a FSimGate(π/2, π/6) gate.

    The unitary of this matrix is

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/6)]]
    """

    def __init__(self):
        super().__init__(theta=np.pi / 2, phi=np.pi / 6)

    def __repr__(self) -> str:
        return 'cirq.SycamoreGate()'

    def __str__(self) -> str:
        return 'SYC'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        return 'SYC', 'SYC'

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, [])


SYC = SycamoreGate()
