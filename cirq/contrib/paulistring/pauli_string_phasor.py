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

from typing import (
    Dict, Hashable, Iterable, Optional, Sequence, Tuple, Type, TypeVar, Union,
    cast
)

from cirq import ops, value, study, extension

from cirq.contrib.paulistring import (
    Pauli,
    CliffordGate,
    PauliString,
    PauliStringGateOperation,
)


T_DESIRED = TypeVar('T_DESIRED')


class PauliStringPhasor(PauliStringGateOperation,
                        ops.CompositeGate,
                        ops.BoundedEffect,
                        ops.ParameterizableEffect,
                        extension.PotentialImplementation[Union[
                            ops.ExtrapolatableEffect,
                            ops.ReversibleEffect]]):
    '''An operation that phases a Pauli string.'''
    def __init__(self,
                 pauli_string: PauliString,
                 *,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the operation.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        If pauli_string is negative, the sign is transferred to the phase.

        Args:
            pauli_string: The PauliString to phase.
            half_turns: Phasing of the Pauli string, in half_turns.
            rads: Phasing of the Pauli string, in radians.
            degs: Phasing of the Pauli string, in degrees.
        """
        half_turns = value.chosen_angle_to_half_turns(
                            half_turns=half_turns,
                            rads=rads,
                            degs=degs)
        if not isinstance(half_turns, value.Symbol):
            half_turns = 1 - (1 - half_turns) % 2
        super().__init__(pauli_string)
        self.half_turns = half_turns

    def _eq_tuple(self) -> Tuple[Hashable, ...]:
        return (PauliStringPhasor, self.pauli_string, self.half_turns)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._eq_tuple() == other._eq_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._eq_tuple())

    def map_qubits(self, qubit_map: Dict[ops.QubitId, ops.QubitId]):
        ps = self.pauli_string.map_qubits(qubit_map)
        return PauliStringPhasor(ps, half_turns=self.half_turns)

    def _with_half_turns(self, half_turns: float) -> 'PauliStringPhasor':
        return PauliStringPhasor(self.pauli_string, half_turns=half_turns)

    def extrapolate_effect(self, factor: float) -> 'PauliStringPhasor':
        if self.is_parameterized():
            raise ValueError("Parameterized. Don't know how to extrapolate.")
        if isinstance(factor, value.Symbol):
            if self.half_turns == 1:
                half_turns = factor
            else:
                raise ValueError("Don't know how to extrapolate by a symbol.")
        else:
            half_turns = 1 - (1 - cast(float, self.half_turns) * factor) % 2
        return self._with_half_turns(half_turns)

    def __pow__(self, power: float) -> 'PauliStringPhasor':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'PauliStringPhasor':
        return self.extrapolate_effect(-1)

    def default_decompose(self, qubits: Sequence[ops.QubitId]) -> ops.OP_TREE:
        if len(self.pauli_string) <= 0:
            return
        any_qubit = qubits[0]
        to_z_ops = tuple(pauli_string_to_z_ops(self.pauli_string))
        xor_decomp = tuple(xor_nonlocal_decompose(self.pauli_string.qubits(),
                                                  any_qubit))
        yield to_z_ops
        yield xor_decomp
        if isinstance(self.half_turns, value.Symbol):
            if self.pauli_string.negated:
                yield ops.X(any_qubit)
            yield ops.RotZGate(half_turns=self.half_turns)(any_qubit)
            if self.pauli_string.negated:
                yield ops.X(any_qubit)
        else:
            if self.pauli_string.negated:
                half_turns = -self.half_turns
            else:
                half_turns = self.half_turns
            yield ops.Z(any_qubit) ** half_turns
        yield ops.inverse_of_invertible_op_tree(xor_decomp)
        yield ops.inverse_of_invertible_op_tree(to_z_ops)

    def text_diagram_info(self, args: ops.TextDiagramInfoArgs
                          ) -> ops.TextDiagramInfo:
        return self.standard_diagram_info(args,
                                          exponent=self.half_turns,
                                          exponent_absorbs_sign=True)

    def trace_distance_bound(self) -> float:
        return ops.RotZGate(half_turns=self.half_turns).trace_distance_bound()

    def try_cast_to(self,
                    desired_type: Type[T_DESIRED],
                    ext: extension.Extensions
                    ) -> Optional[T_DESIRED]:
        if (desired_type in [ops.ExtrapolatableEffect,
                             ops.ReversibleEffect] and
                not self.is_parameterized()):
            return cast(T_DESIRED, self)
        return super().try_cast_to(desired_type, ext)

    def is_parameterized(self) -> bool:
        return isinstance(self.half_turns, value.Symbol)

    def with_parameters_resolved_by(self, param_resolver: study.ParamResolver
                                    ) -> 'PauliStringPhasor':
        return self._with_half_turns(
                        param_resolver.value_of(self.half_turns))

    def __repr__(self):
        return 'PauliStringPhasor({}, half_turns={})'.format(
                    self.pauli_string, self.half_turns)

    def __str__(self):
        return '{}**{}'.format(self.pauli_string, self.half_turns)


def pauli_string_to_z_ops(pauli_string: PauliString) -> Iterable[ops.Operation]:
    """Yields the single qubit operations to apply before a Pauli string of Zs
    (and apply the inverse of these operations after) to make it equivalent to
    the given pauli_string."""
    for qubit, pauli in pauli_string.items():
        yield CliffordGate.from_single_map({pauli: (Pauli.Z, False)})(qubit)


def xor_nonlocal_decompose(qubits: Iterable[ops.QubitId],
                           onto_qubit: ops.QubitId) -> Iterable[ops.Operation]:
    """Decomposition ignores connectivity."""
    for qubit in qubits:
        if qubit != onto_qubit:
            yield ops.CNOT(qubit, onto_qubit)
