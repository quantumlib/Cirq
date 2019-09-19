# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Dict, ItemsView, Iterable, Iterator, KeysView, Mapping,
                    Tuple, TypeVar, Union, ValuesView, overload, Optional, cast)

import cmath
import math
import numbers

import numpy as np

from cirq import value, protocols, linalg
from cirq.ops import (
    global_phase_op,
    raw_types,
    gate_operation,
    common_gates,
    pauli_gates,
    clifford_gate,
    pauli_interaction_gate,
)
TDefault = TypeVar('TDefault')


@value.value_equality(approximate=True, manual_cls=True)
class PauliString(raw_types.Operation):

    def __init__(self,
                 qubit_pauli_map: Optional[
                     Mapping[raw_types.Qid, pauli_gates.Pauli]] = None,
                 coefficient: Union[int, float, complex] = 1) -> None:
        if qubit_pauli_map is None:
            qubit_pauli_map = {}
        qubit_pauli_map = {
            q: p
            for q, p in qubit_pauli_map.items()
            if not isinstance(p, common_gates.IdentityGate)
        }
        for p in qubit_pauli_map.values():
            if not isinstance(p, pauli_gates.Pauli):
                raise TypeError(f'{p} is not a Pauli')
        self._qubit_pauli_map = dict(qubit_pauli_map)
        self._coefficient = complex(coefficient)

    @staticmethod
    def from_single(qubit: raw_types.Qid,
                    pauli: pauli_gates.Pauli) -> 'PauliString':
        """Creates a PauliString with a single qubit."""
        return PauliString({qubit: pauli})

    @property
    def coefficient(self) -> complex:
        return self._coefficient

    def _value_equality_values_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            q, p = list(self._qubit_pauli_map.items())[0]
            return gate_operation.GateOperation(p,
                                                [q])._value_equality_values_()

        return (frozenset(self._qubit_pauli_map.items()),
                self._coefficient)

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            # JSON requires mappings to have string keys.
            'qubit_pauli_map': list(self._qubit_pauli_map.items()),
            'coefficient': self.coefficient,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_pauli_map, coefficient, **kwargs):
        return cls(qubit_pauli_map=dict(qubit_pauli_map),
                   coefficient=coefficient)

    def _value_equality_values_cls_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            return gate_operation.GateOperation
        return PauliString

    def equal_up_to_coefficient(self, other: 'PauliString') -> bool:
        return self._qubit_pauli_map == other._qubit_pauli_map

    def __getitem__(self, key: raw_types.Qid) -> pauli_gates.Pauli:
        return self._qubit_pauli_map[key]

    # pylint: disable=function-redefined
    @overload
    def get(self, key: raw_types.Qid) -> pauli_gates.Pauli:
        pass

    @overload
    def get(self, key: raw_types.Qid,
            default: TDefault) -> Union[pauli_gates.Pauli, TDefault]:
        pass

    def get(self, key: raw_types.Qid, default=None):
        return self._qubit_pauli_map.get(key, default)
    # pylint: enable=function-redefined

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return PauliString(self._qubit_pauli_map,
                               self._coefficient * complex(other))
        if isinstance(other, PauliString):
            s1 = set(self.keys())
            s2 = set(other.keys())
            extra_phase = 1
            terms = {}
            for c in s1 - s2:
                terms[c] = self[c]
            for c in s2 - s1:
                terms[c] = other[c]
            for c in s1 & s2:
                f, p = self[c].phased_pauli_product(other[c])
                extra_phase *= f
                if p != common_gates.I:
                    terms[c] = p
            return PauliString(
                terms, self.coefficient * other.coefficient * extra_phase)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return PauliString(self._qubit_pauli_map,
                               self._coefficient * complex(other))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return PauliString(self._qubit_pauli_map,
                               self._coefficient / complex(other))
        return NotImplemented

    def __add__(self, other):
        from cirq.ops.linear_combinations import PauliSum
        return PauliSum.from_pauli_strings(self).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from cirq.ops.linear_combinations import PauliSum
        return PauliSum.from_pauli_strings(self).__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __contains__(self, key: raw_types.Qid) -> bool:
        return key in self._qubit_pauli_map

    def _decompose_(self):
        if not self._has_unitary_():
            return None
        return [
            *([] if self.coefficient == 1 else
              [global_phase_op.GlobalPhaseOperation(self.coefficient)]),
            *[self[q].on(q) for q in self.qubits],
        ]

    def keys(self) -> KeysView[raw_types.Qid]:
        return self._qubit_pauli_map.keys()

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        return tuple(sorted(self.keys()))

    def with_qubits(self, *new_qubits: raw_types.Qid) -> 'PauliString':
        return PauliString(dict(zip(new_qubits,
                                    (self[q] for q in self.qubits))),
                           self._coefficient)

    def values(self) -> ValuesView[pauli_gates.Pauli]:
        return self._qubit_pauli_map.values()

    def items(self) -> ItemsView:
        return self._qubit_pauli_map.items()

    def __iter__(self) -> Iterator[raw_types.Qid]:
        return iter(self._qubit_pauli_map.keys())

    def __bool__(self):
        return bool(self._qubit_pauli_map)

    def __len__(self) -> int:
        return len(self._qubit_pauli_map)

    def __repr__(self):
        ordered_qubits = sorted(self.qubits)
        prefix = ''

        factors = []
        if self._coefficient == -1:
            prefix = '-'
        elif self._coefficient != 1:
            factors.append(repr(self._coefficient))

        if not ordered_qubits:
            factors.append('cirq.PauliString()')
        for q in ordered_qubits:
            factors.append(repr(cast(raw_types.Gate, self[q]).on(q)))

        fused = prefix + '*'.join(factors)
        if len(factors) > 1:
            return '({})'.format(fused)
        return fused

    def __str__(self):
        ordered_qubits = sorted(self.qubits)
        prefix = ''

        factors = []
        if self._coefficient == -1:
            prefix = '-'
        elif self._coefficient != 1:
            factors.append(repr(self._coefficient))

        if not ordered_qubits:
            factors.append('I')
        for q in ordered_qubits:
            factors.append(str(cast(raw_types.Gate, self[q]).on(q)))

        return prefix + '*'.join(factors)

    def _has_unitary_(self) -> bool:
        return abs(1 - abs(self.coefficient)) < 1e-6

    def _unitary_(self) -> Optional[np.ndarray]:
        if not self._has_unitary_():
            return None
        return linalg.kron(self.coefficient,
                           *[protocols.unitary(self[q]) for q in self.qubits])

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'):
        if not self._has_unitary_():
            return None
        if self.coefficient != 1:
            args.target_tensor *= self.coefficient
        return protocols.apply_unitaries([self[q].on(q) for q in self.qubits],
                                         self.qubits, args)

    def expectation_from_wavefunction(self, state: np.ndarray,
                                      qubit_map: Mapping[raw_types.Qid, int]
                                     ) -> float:
        r"""Evaluate the expectation of this PauliString given a wavefunction.

        Compute the expectation value of this PauliString with respect to a
        wavefunction. By convention expectation values are defined for Hermitian
        operators, and so this method will fail if this PauliString is
        non-Hermitian.

        `state` must be an array representation of a wavefunction and have
        shape `(2 ** n, )` or `(2, 2, ..., 2)` (n entries) where `state` is
        expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state: An array representing a valid wavefunction.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError if this PauliString is non-Hermitian.
        """
        if abs(self.coefficient.imag) > 0.0001:
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                "PauliString <{}>. Coefficient must be real.".format(self))

        # FIXME: Avoid enforce specific complex type. This is necessary to
        # prevent an `apply_unitary` bug (Issue #2041).
        if state.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or "
                            "np.complex128")

        size = state.size
        num_qubits = size.bit_length() - 1
        if len(state.shape) != 1 and state.shape != (2,) * num_qubits:
            raise ValueError("Input array does not represent a wavefunction "
                             "with shape `(2 ** n,)` or `(2, ..., 2)`.")

        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        # HACK: avoid circular import
        from cirq.sim.wave_function import validate_normalized_state
        validate_normalized_state(state=state,
                                  qid_shape=(2,) * num_qubits,
                                  dtype=state.dtype)
        return self._expectation_from_wavefunction_no_validation(
            state, qubit_map)

    def _expectation_from_wavefunction_no_validation(
            self, state: np.ndarray,
            qubit_map: Mapping[raw_types.Qid, int]) -> float:
        """Evaluate the expectation of this PauliString given a wavefunction.

        This method does not provide input validation. See
        `PauliString.expectation_from_wavefunction` for function description.

        Args:
            state: An array representing a valid wavefunction.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        if len(state.shape) == 1:
            num_qubits = state.shape[0].bit_length() - 1
            state = np.reshape(state, (2,) * num_qubits)

        ket = np.copy(state)
        for qubit, pauli in self.items():
            buffer = np.empty(ket.shape, dtype=state.dtype)
            args = protocols.ApplyUnitaryArgs(target_tensor=ket,
                                              available_buffer=buffer,
                                              axes=(qubit_map[qubit],))
            ket = protocols.apply_unitary(pauli, args)

        return self.coefficient * (np.tensordot(
            state.conj(), ket, axes=len(ket.shape)).item())

    def expectation_from_density_matrix(self, state: np.ndarray,
                                        qubit_map: Mapping[raw_types.Qid, int]
                                       ) -> float:
        r"""Evaluate the expectation of this PauliString given a density matrix.

        Compute the expectation value of this PauliString with respect to an
        array representing a density matrix. By convention expectation values
        are defined for Hermitian operators, and so this method will fail if
        this PauliString is non-Hermitian.

        `state` must be an array representation of a density matrix and have
        shape `(2 ** n, 2 ** n)` or `(2, 2, ..., 2)` (2*n entries), where
        `state` is expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError if this PauliString is non-Hermitian.
        """
        if abs(self.coefficient.imag) > 0.0001:
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                "PauliString <{}>. Coefficient must be real.".format(self))

        # FIXME: Avoid enforcing specific complex type. This is necessary to
        # prevent an `apply_unitary` bug (Issue #2041).
        if state.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or "
                            "np.complex128")

        size = state.size
        num_qubits = int(np.sqrt(size)).bit_length() - 1
        dim = 1 << num_qubits
        if state.shape != (dim, dim) and state.shape != (2, 2) * num_qubits:
            raise ValueError("Input array does not represent a density matrix "
                             "with shape `(2 ** n, 2 ** n)` or `(2, ..., 2)`.")

        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        # HACK: avoid circular import
        from cirq.sim.density_matrix_utils import to_valid_density_matrix
        # Do not enforce reshaping if the state all axes are dimension 2.
        _ = to_valid_density_matrix(density_matrix_rep=state.reshape(dim, dim),
                                    num_qubits=num_qubits,
                                    dtype=state.dtype)
        return self._expectation_from_density_matrix_no_validation(
            state, qubit_map)

    def _expectation_from_density_matrix_no_validation(
            self, state: np.ndarray,
            qubit_map: Mapping[raw_types.Qid, int]) -> float:
        """Evaluate the expectation of this PauliString given a density matrix.

        This method does not provide input validation. See
        `PauliString.expectation_from_density_matrix` for function description.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        result = np.copy(state)
        if len(state.shape) == 2:
            num_qubits = state.shape[0].bit_length() - 1
            result = np.reshape(result, (2,) * num_qubits * 2)

        for qubit, pauli in self.items():
            buffer = np.empty(result.shape, dtype=state.dtype)
            args = protocols.ApplyUnitaryArgs(target_tensor=result,
                                              available_buffer=buffer,
                                              axes=(qubit_map[qubit],))
            result = protocols.apply_unitary(pauli, args)

        while any(result.shape):
            result = np.trace(result, axis1=0, axis2=len(result.shape) // 2)
        return result * self.coefficient

    def zip_items(self, other: 'PauliString') -> Iterator[
            Tuple[raw_types.Qid, Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]]:
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield qubit, (pauli0, other[qubit])

    def zip_paulis(self, other: 'PauliString'
                  ) -> Iterator[Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]:
        return (paulis for qubit, paulis in self.zip_items(other))

    def commutes_with(self, other: 'PauliString') -> bool:
        return sum(not p0.commutes_with(p1)
                   for p0, p1 in self.zip_paulis(other)
                   ) % 2 == 0

    def __neg__(self) -> 'PauliString':
        return PauliString(self._qubit_pauli_map, -self._coefficient)

    def __pos__(self) -> 'PauliString':
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override behavior of numpy's exp method."""
        if ufunc == np.exp and len(inputs) == 1 and inputs[0] is self:
            return math.e**self
        return NotImplemented

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PauliString(self._qubit_pauli_map, self.coefficient**-1)
        if isinstance(power, (int, float)):
            r, i = cmath.polar(self.coefficient)
            if abs(r - 1) > 0.0001:
                # Raising non-unitary PauliStrings to a power is not supported.
                return NotImplemented

            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {
                    pauli_gates.X: common_gates.XPowGate,
                    pauli_gates.Y: common_gates.YPowGate,
                    pauli_gates.Z: common_gates.ZPowGate,
                }
                return gates[p](exponent=power).on(q)

            global_half_turns = power * (i / math.pi)

            # HACK: Avoid circular dependency.
            from cirq.ops import pauli_string_phasor
            return pauli_string_phasor.PauliStringPhasor(
                PauliString(self._qubit_pauli_map),
                exponent_neg=global_half_turns + power,
                exponent_pos=global_half_turns)
        return NotImplemented

    def __rpow__(self, base):
        if isinstance(base, (int, float)) and base > 0:
            if abs(self.coefficient.real) > 0.0001:
                raise NotImplementedError(
                    "Exponentiated to a non-Hermitian PauliString <{}**{}>. "
                    "Coefficient must be imaginary.".format(base, self))

            half_turns = math.log(base) * (-self.coefficient.imag / math.pi)

            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {
                    pauli_gates.X: common_gates.XPowGate,
                    pauli_gates.Y: common_gates.YPowGate,
                    pauli_gates.Z: common_gates.ZPowGate,
                }
                return gates[p](exponent=half_turns, global_shift=-0.5).on(q)

            # HACK: Avoid circular dependency.
            from cirq.ops import pauli_string_phasor
            return pauli_string_phasor.PauliStringPhasor(
                PauliString(self._qubit_pauli_map),
                exponent_neg=+half_turns / 2,
                exponent_pos=-half_turns / 2)
        return NotImplemented

    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]
                   ) -> 'PauliString':
        new_qubit_pauli_map = {qubit_map[qubit]: pauli
                               for qubit, pauli in self.items()}
        return PauliString(new_qubit_pauli_map, self._coefficient)

    def to_z_basis_ops(self) -> Iterator[raw_types.Operation]:
        """Returns operations to convert the qubits to the computational basis.
        """
        for qubit, pauli in self.items():
            yield clifford_gate.SingleQubitCliffordGate.from_single_map(
                {pauli: (pauli_gates.Z, False)})(qubit)

    def pass_operations_over(self,
                             ops: Iterable[raw_types.Operation],
                             after_to_before: bool = False) -> 'PauliString':
        """Determines how the Pauli string changes when conjugated by Cliffords.

        The output and input pauli strings are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_STRING───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_STRING───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
        pauli_map = dict(self._qubit_pauli_map)
        should_negate = False
        for op in ops:
            if not set(op.qubits) & set(pauli_map.keys()):
                # op operates on an independent set of qubits from the Pauli
                # string.  The order can be switched with no change no matter
                # what op is.
                continue
            should_negate ^= PauliString._pass_operation_over(pauli_map,
                                                              op,
                                                              after_to_before)
        coef = -self._coefficient if should_negate else self.coefficient
        return PauliString(pauli_map, coef)

    @staticmethod
    def _pass_operation_over(pauli_map: Dict[raw_types.Qid, pauli_gates.Pauli],
                             op: raw_types.Operation,
                             after_to_before: bool = False) -> bool:
        if isinstance(op, gate_operation.GateOperation):
            gate = op.gate
            if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
                return PauliString._pass_single_clifford_gate_over(
                    pauli_map, gate, op.qubits[0],
                    after_to_before=after_to_before)
            if isinstance(gate, common_gates.CZPowGate):
                gate = pauli_interaction_gate.PauliInteractionGate.CZ
            if isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
                return PauliString._pass_pauli_interaction_gate_over(
                    pauli_map, gate, op.qubits[0], op.qubits[1],
                    after_to_before=after_to_before)
        raise TypeError('Unsupported operation: {!r}'.format(op))

    @staticmethod
    def _pass_single_clifford_gate_over(
            pauli_map: Dict[raw_types.Qid, pauli_gates.Pauli],
            gate: clifford_gate.SingleQubitCliffordGate,
            qubit: raw_types.Qid,
            after_to_before: bool = False) -> bool:
        if qubit not in pauli_map:
            return False
        if not after_to_before:
            gate **= -1
        pauli, inv = gate.transform(pauli_map[qubit])
        pauli_map[qubit] = pauli
        return inv

    @staticmethod
    def _pass_pauli_interaction_gate_over(
            pauli_map: Dict[raw_types.Qid, pauli_gates.Pauli],
            gate: pauli_interaction_gate.PauliInteractionGate,
            qubit0: raw_types.Qid,
            qubit1: raw_types.Qid,
            after_to_before: bool = False) -> bool:

        def merge_and_kickback(qubit: raw_types.Qid,
                               pauli_left: Optional[pauli_gates.Pauli],
                               pauli_right: Optional[pauli_gates.Pauli],
                               inv: bool) -> int:
            assert pauli_left is not None or pauli_right is not None
            if pauli_left is None or pauli_right is None:
                pauli_map[qubit] = cast(pauli_gates.Pauli,
                                        pauli_left or pauli_right)
                return 0
            if pauli_left == pauli_right:
                del pauli_map[qubit]
                return 0

            pauli_map[qubit] = pauli_left.third(pauli_right)
            if (pauli_left < pauli_right) ^ after_to_before:
                return int(inv) * 2 + 1

            return int(inv) * 2 - 1

        quarter_kickback = 0
        if (qubit0 in pauli_map and
                not pauli_map[qubit0].commutes_with(gate.pauli0)):
            quarter_kickback += merge_and_kickback(qubit1,
                                                   gate.pauli1,
                                                   pauli_map.get(qubit1),
                                                   gate.invert1)
        if (qubit1 in pauli_map and
                not pauli_map[qubit1].commutes_with(gate.pauli1)):
            quarter_kickback += merge_and_kickback(qubit0,
                                                   pauli_map.get(qubit0),
                                                   gate.pauli0,
                                                   gate.invert0)
        assert quarter_kickback % 2 == 0, (
            'Impossible condition.  '
            'quarter_kickback is either incremented twice or never.')
        return quarter_kickback % 4 == 2


def _validate_qubit_mapping(qubit_map: Mapping[raw_types.Qid, int],
                            pauli_qubits: Tuple[raw_types.Qid, ...],
                            num_state_qubits: int) -> None:
    """Validates that a qubit map is a valid mapping.

    This will enforce that all elements of `pauli_qubits` appear in `qubit_map`,
    and that the integers in `qubit_map` correspond to valid positions in a
    representation of a state over `num_state_qubits`.

    Args:
        qubit_map: A map from qubits to integers.
        pauli_qubits: The qubits that must be contained in `qubit_map`.
        num_state_qubits: The number of qubits over which a state is expressed.
    """
    if not isinstance(qubit_map, Mapping) or not all(
            isinstance(k, raw_types.Qid) and isinstance(v, int)
            for k, v in qubit_map.items()):
        raise TypeError("Input qubit map must be a valid mapping from "
                        "Qubit ID's to integer indices.")

    if not set(qubit_map.keys()) >= set(pauli_qubits):
        raise ValueError("Input qubit map must be a complete mapping over all "
                         " of this PauliString's qubits.")

    used_inds = [qubit_map[q] for q in pauli_qubits]
    if len(used_inds) != len(set(used_inds)) or not set(
            range(num_state_qubits)) >= set(sorted(used_inds)):
        raise ValueError("Input qubit map indices must be valid for a state "
                         "over {} qubits.".format(num_state_qubits))


# Ignoring type because mypy believes `with_qubits` methods are incompatible.
class SingleQubitPauliStringGateOperation(  # type: ignore
        gate_operation.GateOperation, PauliString):
    """A Pauli operation applied to a qubit.

    Satisfies the contract of both GateOperation and PauliString. Relies
    implicitly on the fact that PauliString({q: X}) compares as equal to
    GateOperation(X, [q]).
    """

    def __init__(self, pauli: pauli_gates.Pauli, qubit: raw_types.Qid):
        PauliString.__init__(self, {qubit: pauli})
        gate_operation.GateOperation.__init__(self, cast(raw_types.Gate, pauli),
                                              [qubit])

    def with_qubits(self, *new_qubits: raw_types.Qid
                   ) -> 'SingleQubitPauliStringGateOperation':
        if len(new_qubits) != 1:
            raise ValueError("len(new_qubits) != 1")
        return SingleQubitPauliStringGateOperation(
            cast(pauli_gates.Pauli, self.gate), new_qubits[0])

    @property
    def pauli(self) -> pauli_gates.Pauli:
        return cast(pauli_gates.Pauli, self.gate)

    @property
    def qubit(self) -> raw_types.Qid:
        assert len(self.qubits) == 1
        return self.qubits[0]

    def _as_pauli_string(self) -> PauliString:
        return PauliString({self.qubit: self.pauli})

    def __mul__(self, other):
        if isinstance(other, SingleQubitPauliStringGateOperation):
            return self._as_pauli_string() * other._as_pauli_string()
        if isinstance(other, (PauliString, complex, float, int)):
            return self._as_pauli_string() * other
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (PauliString, complex, float, int)):
            return other * self._as_pauli_string()
        return NotImplemented

    def __neg__(self):
        return -self._as_pauli_string()

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['pauli', 'qubit'])

    @classmethod
    def _from_json_dict_(  # type: ignore
            cls, pauli: pauli_gates.Pauli, qubit: raw_types.Qid, **kwargs):
        # Note, this method is required or else superclasses' deserialization
        # would be used
        return cls(pauli=pauli, qubit=qubit)
