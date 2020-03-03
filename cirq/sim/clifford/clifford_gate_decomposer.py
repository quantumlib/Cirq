import numpy as np
from cirq import protocols, ops

import cirq

from typing import (Tuple)

_ALL_CLIFFORD_GATES = [
    '', # I
    'HSSH', # =X
    'HSSSH',
    'HSH',
    'SS', # =Z
    'HSSHSS', # = -i*Y
    'SHSSS',
    'SSSHS',
    'HSSS',
    'SHSSSH',
    'S',
    'HSSHSSS',
    'HS',
    'SSHSSS',
    'SSS',
    'HSSHS',
    'H',
    'SSH',
    'SH',
    'HSHS',
    'HSS',
    'SSHSS',
    'SHSS',
    'SSSHSS'
]

class CliffordGateDecomposer:
    def __init__(self):
        s = protocols.unitary(ops.S)
        h = protocols.unitary(ops.H)

        # Converts string made of letters S and H to gate.
        def name_to_unitary(name):
            result = np.eye(2)
            for ch in name:
                if ch == 'S': result = result @ s
                elif ch == 'H': result = result @ h
            return result

        self._unitaries = [(name, name_to_unitary(name)) for name in _ALL_CLIFFORD_GATES]

    """
    Decomposes unitary into SH gates and global phase.
    Raises ValueError if it's not possible.
    """
    def decompose(self, gate : 'cirq.Gate') -> Tuple[str, np.complex128]:
        unitary = protocols.unitary(gate)
        assert unitary.shape == (2, 2)

        def _possible_phase_shift(matrix1, matrix2):
            if np.abs(matrix2[0, 0]) > 1e-9:
                return matrix1[0, 0] / matrix2[0, 0]
            else:
                return matrix1[0, 1] / matrix2[0, 1]

        for name, unitary_candidate in self._unitaries:
            phase_shift = _possible_phase_shift(unitary, unitary_candidate)
            if np.allclose(unitary, unitary_candidate * phase_shift):
                return name, phase_shift


        raise ValueError('%s cannot be run with Clifford simulator' %
                        str(gate))  # type: ignore

CLIFFORD_GATE_DECOMPOSER = CliffordGateDecomposer()
