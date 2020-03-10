from typing import (Tuple)

import numpy as np

from cirq import protocols, ops


def _name_to_unitary(name):
    """Converts string made of letters S and H to gate."""
    result = np.eye(2)
    for ch in name:
        if ch == 'S':
            result = result @ protocols.unitary(ops.S)
        elif ch == 'H':
            result = result @ protocols.unitary(ops.H)
    return result

class CliffordGateDecomposer:
    # All 24 Clifford gates, written as products of H and S gates.
    _all_gates = [
        '', 'HSSH', 'HSSSH', 'HSH', 'SS', 'HSSHSS', 'SHSSS', 'SSSHS', 'HSSS',
        'SHSSSH', 'S', 'HSSHSSS', 'HS', 'SSHSSS', 'SSS', 'HSSHS', 'H', 'SSH',
        'SH', 'HSHS', 'HSS', 'SSHSS', 'SHSS', 'SSSHSS'
    ]

    _candidates = [(name, _name_to_unitary(name)) for name in _all_gates]

    @staticmethod
    def decompose(gate: 'ops.Gate') -> Tuple[str, np.complex128]:
        """Decomposes unitary into S or H gates and global phase.

        Args:
            gate: Clifford gate to decompose.
        Returns:
            Tuple of string and phase (complex number with absolute value 1).
            String consists exclusively of letters S or H.
            Given gate is equal to product of S and H gates as specified in the
            string, multiplied by the phase.

        Raises ValueError if `gate` is not a Clifford gate.
        """
        unitary = protocols.unitary(gate)
        assert unitary.shape == (2, 2)

        def _possible_phase_shift(matrix1, matrix2):
            if np.abs(matrix2[0, 0]) > 1e-9:
                return matrix1[0, 0] / matrix2[0, 0]
            else:
                return matrix1[0, 1] / matrix2[0, 1]

        for name, candidate in CliffordGateDecomposer._candidates:
            phase_shift = _possible_phase_shift(unitary, candidate)
            if np.allclose(unitary, candidate * phase_shift):
                return name, phase_shift

        raise ValueError('%s is not a Clifford gate.' % str(gate))

    @staticmethod
    def can_decompose(gate: 'ops.Gate') -> bool:
        """Checks whether this gate can be decomposed."""
        try:
            CliffordGateDecomposer.decompose(gate)
            return True
        except ValueError:
            return False
