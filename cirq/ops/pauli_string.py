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
                    Tuple, TypeVar, Union, ValuesView, overload, Optional, cast,
                    TYPE_CHECKING, SupportsComplex, List)

import cmath
import math
import numbers

import numpy as np

from cirq import value, protocols, linalg
from cirq._compat import deprecated
from cirq.ops import (
    global_phase_op,
    raw_types,
    gate_operation,
    common_gates,
    pauli_gates,
    clifford_gate,
    pauli_interaction_gate,
    identity,
)

if TYPE_CHECKING:
    import cirq

# A value that can be unambiguously converted into a `cirq.PauliString`.
PAULI_STRING_LIKE = Union[
    complex, 'cirq.OP_TREE',
    Mapping['cirq.Qid', Union['cirq.Pauli', 'cirq.IdentityGate']],
    Iterable,  # of PAULI_STRING_LIKE, but mypy doesn't do recursive types yet.
]

TDefault = TypeVar('TDefault')


@value.value_equality(approximate=True, manual_cls=True)
class PauliString(raw_types.Operation):

    def __init__(
            self,
            *contents: PAULI_STRING_LIKE,
            qubit_pauli_map: Optional[Dict['cirq.Qid', 'cirq.Pauli']] = None,
            coefficient: Union[int, float, complex] = 1):
        """Initializes a new PauliString.

        Args:
            *contents: A value or values to convert into a pauli string. This
                can be a number, a pauli operation, a dictionary from qubit to
                pauli/identity gates, or collections thereof. If a list of
                values is given, they are each individually converted and then
                multiplied from left to right in order.
            qubit_pauli_map: Initial dictionary mapping qubits to pauli
                operations. Defaults to the empty dictionary. Note that, unlike
                dictionaries passed to contents, this dictionary must not
                contain any identity gate values. Further note that this
                argument specifies values that are logically *before* factors
                specified in `contents`; `contents` are *right* multiplied onto
                the values in this dictionary.
            coefficient: Initial scalar coefficient. Defaults to 1.

        Examples:
            >>> a, b, c = cirq.LineQubit.range(3)

            >>> print(cirq.PauliString([cirq.X(a), cirq.X(a)]))
            I

            >>> print(cirq.PauliString(-1, cirq.X(a), cirq.Y(b), cirq.Z(c)))
            -X(0)*Y(1)*Z(2)

            >>> print(cirq.PauliString({a: cirq.X}, [-2, 3, cirq.Y(a)]))
            -6j*Z(0)

            >>> print(cirq.PauliString({a: cirq.I, b: cirq.X}))
            X(1)

            >>> print(cirq.PauliString({a: cirq.Y},
            ...                        qubit_pauli_map={a: cirq.X}))
            1j*Z(0)
        """
        if qubit_pauli_map is not None:
            for v in qubit_pauli_map.values():
                if not isinstance(v, pauli_gates.Pauli):
                    raise TypeError(f'{v} is not a Pauli')

        p = _MutablePauliString(coef=complex(coefficient),
                                paulis=dict(qubit_pauli_map or {}))
        p.inline_times_pauli_string_like(contents)
        self._qubit_pauli_map = p.paulis
        self._coefficient = p.coef

    @staticmethod
    @deprecated(deadline="v0.7.0",
                fix="call cirq.PauliString(pauli(qubit)) instead")
    def from_single(qubit: raw_types.Qid,
                    pauli: pauli_gates.Pauli) -> 'PauliString':
        """Creates a PauliString with a single qubit."""
        return PauliString(qubit_pauli_map={qubit: pauli})

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

    def __mul__(self, other) -> 'PauliString':
        if not isinstance(
                other,
            (PauliString, numbers.Number, identity.IdentityOperation)):
            return NotImplemented

        return PauliString(cast(PAULI_STRING_LIKE, other),
                           qubit_pauli_map=self._qubit_pauli_map,
                           coefficient=self.coefficient)

    @property
    def gate(self) -> 'cirq.DensePauliString':
        order: List[Optional[pauli_gates.Pauli]] = [
            None, pauli_gates.X, pauli_gates.Y, pauli_gates.Z
        ]
        from cirq.ops.dense_pauli_string import DensePauliString
        return DensePauliString(
            coefficient=self.coefficient,
            pauli_mask=[order.index(self[q]) for q in self.qubits])

    def __rmul__(self, other) -> 'PauliString':
        if isinstance(other, numbers.Number):
            return PauliString(qubit_pauli_map=self._qubit_pauli_map,
                               coefficient=self._coefficient *
                               complex(cast(SupportsComplex, other)))

        if isinstance(other, identity.IdentityOperation):
            return self

        # Note: PauliString case handled by __mul__.
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return PauliString(qubit_pauli_map=self._qubit_pauli_map,
                               coefficient=self._coefficient /
                               complex(cast(SupportsComplex, other)))
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
        return PauliString(qubit_pauli_map=dict(
            zip(new_qubits, (self[q] for q in self.qubits))),
                           coefficient=self._coefficient)

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

    def expectation_from_wavefunction(self,
                                      state: np.ndarray,
                                      qubit_map: Mapping[raw_types.Qid, int],
                                      *,
                                      atol: float = 1e-7,
                                      check_preconditions: bool = True
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
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid wavefunction.

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
        if check_preconditions:
            # HACK: avoid circular import
            from cirq.sim.wave_function import validate_normalized_state
            validate_normalized_state(state=state,
                                      qid_shape=(2,) * num_qubits,
                                      dtype=state.dtype,
                                      atol=atol)
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

    def expectation_from_density_matrix(self,
                                        state: np.ndarray,
                                        qubit_map: Mapping[raw_types.Qid, int],
                                        *,
                                        atol: float = 1e-7,
                                        check_preconditions: bool = True
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
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid density matrix.

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
        if check_preconditions:
            # HACK: avoid circular import
            from cirq.sim.density_matrix_utils import to_valid_density_matrix
            # Do not enforce reshaping if the state all axes are dimension 2.
            _ = to_valid_density_matrix(density_matrix_rep=state.reshape(
                dim, dim),
                                        num_qubits=num_qubits,
                                        dtype=state.dtype,
                                        atol=atol)
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
        return PauliString(qubit_pauli_map=self._qubit_pauli_map,
                           coefficient=-self._coefficient)

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
            return PauliString(qubit_pauli_map=self._qubit_pauli_map,
                               coefficient=self.coefficient**-1)
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
                PauliString(qubit_pauli_map=self._qubit_pauli_map),
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
                PauliString(qubit_pauli_map=self._qubit_pauli_map),
                exponent_neg=+half_turns / 2,
                exponent_pos=-half_turns / 2)
        return NotImplemented

    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]
                   ) -> 'PauliString':
        new_qubit_pauli_map = {qubit_map[qubit]: pauli
                               for qubit, pauli in self.items()}
        return PauliString(qubit_pauli_map=new_qubit_pauli_map,
                           coefficient=self._coefficient)

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
        return PauliString(qubit_pauli_map=pauli_map, coefficient=coef)

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
        return PauliString(qubit_pauli_map={self.qubit: self.pauli})

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


class _MutablePauliString:

    def __init__(self, *, coef: complex,
                 paulis: Dict['cirq.Qid', 'cirq.Pauli']):
        self.coef = coef
        self.paulis = paulis

    def _inline_times_pauli(self, qubit: 'cirq.Qid', pauli: 'cirq.Pauli'):
        cur_pauli = self.paulis.get(qubit, None)
        if cur_pauli is None:
            self.paulis[qubit] = pauli
            return

        phase, new_pauli = cur_pauli.phased_pauli_product(pauli)
        self.coef *= phase
        if new_pauli is identity.I:
            del self.paulis[qubit]
        else:
            self.paulis[qubit] = cast(pauli_gates.Pauli, new_pauli)

    def inline_times_pauli_string(self, other: 'cirq.PauliString'):
        for qubit, pauli in other.items():
            self._inline_times_pauli(qubit, pauli)
        self.coef *= other.coefficient

    def _inline_times_mapping(
            self, mapping: Mapping['cirq.Qid',
                                   Union['cirq.Pauli', 'cirq.IdentityGate']]):
        for qubit, pauli in mapping.items():
            if isinstance(pauli, identity.IdentityGate):
                continue

            if not isinstance(pauli, pauli_gates.Pauli):
                raise TypeError(
                    f'{repr(pauli)} is not a Pauli or identity gate.')

            self._inline_times_pauli(qubit, pauli)

    def inline_times_pauli_string_like(self,
                                       contents: 'cirq.PAULI_STRING_LIKE'):
        if isinstance(contents, PauliString):
            # Note: cirq.X/Y/Z(qubit) are PauliString instances.
            self.inline_times_pauli_string(contents)
        elif isinstance(contents, identity.IdentityOperation):
            pass  # No effect.
        elif isinstance(contents, Mapping):
            self._inline_times_mapping(contents)
        elif isinstance(contents, Iterable) and not isinstance(contents, str):
            for item in contents:
                self.inline_times_pauli_string_like(
                    cast(PAULI_STRING_LIKE, item))
        elif isinstance(contents, numbers.Number):
            self.coef *= complex(cast(SupportsComplex, contents))
        else:
            raise TypeError(f"Not a `cirq.PAULI_STRING_LIKE`: "
                            f"{type(contents)}, {repr(contents)}")
