import numbers
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy

from cirq import value, ops, protocols, linalg
from cirq.ops import gate_features
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PhasedXZGate(gate_features.SingleQubitGate):
    """A single qubit operation expressed as $Z^z Z^a X^x Z^{-a}$.

    The above expression is a matrix multiplication with time going to the left.
    In quantum circuit notation, this operation decomposes into this circuit:

    ───Z^(-a)──X^x──Z^a────Z^z───$

    The axis phase exponent (a) decides which axis in the XY plane to rotate
    around. The amount of rotation around that axis is decided by the x
    exponent (x). Then the z exponent (z) decides how much to phase the qubit.
    """

    def __init__(self, *, x_exponent: Union[numbers.Real, sympy.Basic],
                 z_exponent: Union[numbers.Real, sympy.Basic],
                 axis_phase_exponent: Union[numbers.Real, sympy.Basic]) -> None:
        """
        Args:
            x_exponent: Determines how much to rotate during the
                axis-in-XY-plane rotation. The $x$ in $Z^z Z^a X^x Z^{-a}$.
            z_exponent: The amount of phasing to apply after the
                axis-in-XY-plane rotation. The $z$ in $Z^z Z^a X^x Z^{-a}$.
            axis_phase_exponent: Determines which axis to rotate around during
                the axis-in-XY-plane rotation. The $a$ in $Z^z Z^a X^x Z^{-a}$.
        """
        self._x_exponent = x_exponent
        self._z_exponent = z_exponent
        self._axis_phase_exponent = axis_phase_exponent

    def _canonical(self) -> 'cirq.PhasedXZGate':
        x = self.x_exponent
        z = self.z_exponent
        a = self.axis_phase_exponent

        # Canonicalize X exponent into (-1, +1].
        if isinstance(x, numbers.Real):
            x %= 2
            if x > 1:
                x -= 2

        # Axis phase exponent is irrelevant if there is no X exponent.
        if x == 0:
            a = 0
        # For 180 degree X rotations, the axis phase and z exponent overlap.
        if x == 1 and z != 0:
            a += z / 2
            z = 0

        # Canonicalize Z exponent into (-1, +1].
        if isinstance(z, numbers.Real):
            z %= 2
            if z > 1:
                z -= 2

        # Canonicalize axis phase exponent into (-0.5, +0.5].
        if isinstance(a, numbers.Real):
            a %= 2
            if a > 1:
                a -= 2
            if a <= -0.5:
                a += 1
                if x != 1:
                    x = -x
            elif a > 0.5:
                a -= 1
                if x != 1:
                    x = -x

        return PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    @property
    def x_exponent(self) -> Union[numbers.Real, sympy.Basic]:
        return self._x_exponent

    @property
    def z_exponent(self) -> Union[numbers.Real, sympy.Basic]:
        return self._z_exponent

    @property
    def axis_phase_exponent(self) -> Union[numbers.Real, sympy.Basic]:
        return self._axis_phase_exponent

    def _value_equality_values_(self):
        c = self._canonical()
        return (
            value.PeriodicValue(c._x_exponent, 2),
            value.PeriodicValue(c._z_exponent, 2),
            value.PeriodicValue(c._axis_phase_exponent, 2),
        )

    @staticmethod
    def from_matrix(mat: np.array) -> 'cirq.PhasedXZGate':
        pre_phase, rotation, post_phase = (
            linalg.deconstruct_single_qubit_matrix_into_angles(mat))
        pre_phase /= np.pi
        post_phase /= np.pi
        rotation /= np.pi
        pre_phase -= 0.5
        post_phase += 0.5
        return PhasedXZGate(x_exponent=rotation,
                            axis_phase_exponent=-pre_phase,
                            z_exponent=post_phase + pre_phase)._canonical()

    def with_z_exponent(self, z_exponent: Union[numbers.Real, sympy.Basic]
                       ) -> 'cirq.PhasedXZGate':
        return PhasedXZGate(axis_phase_exponent=self._axis_phase_exponent,
                            x_exponent=self._x_exponent,
                            z_exponent=z_exponent)

    def _qasm_(self, args: 'cirq.QasmArgs',
               qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        from cirq.circuits import qasm_output
        qasm_gate = qasm_output.QasmUGate(lmda=0.5 - self._axis_phase_exponent,
                                          theta=self._x_exponent,
                                          phi=self._z_exponent +
                                          self._axis_phase_exponent - 0.5)
        return protocols.qasm(qasm_gate, args=args, qubits=qubits)

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        q = qubits[0]
        yield ops.Z(q)**-self._axis_phase_exponent
        yield ops.X(q)**self._x_exponent
        yield ops.Z(q)**(self._axis_phase_exponent + self._z_exponent)

    def __pow__(self, exponent: Union[float, int]) -> 'PhasedXZGate':
        if exponent == 1:
            return self
        if exponent == -1:
            return PhasedXZGate(
                x_exponent=-self._x_exponent,
                z_exponent=-self._z_exponent,
                axis_phase_exponent=self._z_exponent + self.axis_phase_exponent,
            )
        return NotImplemented

    def _is_parameterized_(self) -> bool:
        """See `cirq.SupportsParameterization`."""
        return (protocols.is_parameterized(self._x_exponent) or
                protocols.is_parameterized(self._z_exponent) or
                protocols.is_parameterized(self._axis_phase_exponent))

    def _resolve_parameters_(self, param_resolver) -> 'cirq.PhasedXZGate':
        """See `cirq.SupportsParameterization`."""
        return PhasedXZGate(
            z_exponent=param_resolver.value_of(self._z_exponent),
            x_exponent=param_resolver.value_of(self._x_exponent),
            axis_phase_exponent=param_resolver.value_of(
                self._axis_phase_exponent),
        )

    def _phase_by_(self, phase_turns, qubit_index) -> 'cirq.PhasedXZGate':
        """See `cirq.SupportsPhase`."""
        assert qubit_index == 0
        return PhasedXZGate(x_exponent=self._x_exponent,
                            z_exponent=self._z_exponent,
                            axis_phase_exponent=self._axis_phase_exponent +
                            phase_turns * 2)

    def _pauli_expansion_(self) -> 'cirq.LinearDict[str]':
        if protocols.is_parameterized(self):
            return NotImplemented
        x_angle = np.pi * self._x_exponent / 2
        z_angle = np.pi * self._z_exponent / 2
        axis_angle = np.pi * self._axis_phase_exponent
        phase = np.exp(1j * (x_angle + z_angle))

        cx = np.cos(x_angle)
        sx = np.sin(x_angle)
        return value.LinearDict({
            'I': phase * cx * np.cos(z_angle),
            'X': -1j * phase * sx * np.cos(z_angle + axis_angle),
            'Y': -1j * phase * sx * np.sin(z_angle + axis_angle),
            'Z': -1j * phase * cx * np.sin(z_angle),
        })  # yapf: disable

    def _circuit_diagram_info_(self,
                               args: 'cirq.CircuitDiagramInfoArgs') -> str:
        """See `cirq.SupportsCircuitDiagramInfo`."""
        return (f'PhXZ('
                f'a={args.format_real(self._axis_phase_exponent)},'
                f'x={args.format_real(self._x_exponent)},'
                f'z={args.format_real(self._z_exponent)})')

    def __str__(self) -> str:
        return protocols.circuit_diagram_info(self).wire_symbols[0]

    def __repr__(self) -> str:
        return (f'cirq.PhasedXZGate('
                f'axis_phase_exponent={proper_repr(self._axis_phase_exponent)},'
                f' x_exponent={proper_repr(self._x_exponent)}, '
                f'z_exponent={proper_repr(self._z_exponent)})')

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(
            self, ['axis_phase_exponent', 'x_exponent', 'z_exponent'])
