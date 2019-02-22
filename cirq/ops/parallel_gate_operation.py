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
    Optional, Sequence, FrozenSet, Tuple, Union, TYPE_CHECKING,
    Any)

from fractions import Fraction
import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types, gate_features, op_tree
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List


@value.value_equality
class ParallelGateOperation(raw_types.Operation):
    """An application of several copies of a gate to a group of qubits."""

    def __init__(self,
                 gate: raw_types.Gate,
                 qubits: Sequence[raw_types.QubitId]) -> None:
        """
        Args:
            gate: the gate to apply
            qubits: lists of lists of qubits to apply the gate to.
        """
        if not isinstance(gate, gate_features.SingleQubitGate):
            raise ValueError("gate must be a single qubit gate")
        if len(frozenset(qubits)) != len(qubits):
            raise ValueError("repeated qubits are not allowed")
        for qubit in qubits:
            gate.validate_args([qubit])
        self._gate = gate
        self._qubits = tuple(qubits)

    @property
    def gate(self) -> raw_types.Gate:
        """The single qubit gate applied by the operation."""
        return self._gate

    @property
    def qubits(self) -> Tuple[raw_types.QubitId, ...]:
        """The qubits targeted by the operation."""
        return self._qubits

    def with_qubits(self,
                    *new_qubits: raw_types.QubitId) -> 'ParallelGateOperation':
        """ParallelGateOperation with same the gate but new qubits"""
        return ParallelGateOperation(self.gate, new_qubits)

    def with_gate(self, new_gate: raw_types.Gate) -> 'ParallelGateOperation':
        """ParallelGateOperation with same qubits but a new gate"""
        return ParallelGateOperation(new_gate, self.qubits)

    def __repr__(self):
        return 'cirq.ParallelGateOperation(gate={!r}, qubits={!r})'.format(
            self.gate,
            list(self.qubits))

    def __str__(self):
        return '{}({})'.format(self.gate,
                               ', '.join(str(e) for e in self.qubits))

    def _group_interchangeable_qubits(self) -> Tuple[
            Union[raw_types.QubitId,
                  Tuple[int, FrozenSet[raw_types.QubitId]]],
            ...]:
        return tuple(((0, frozenset(self.qubits)),))

    def _value_equality_values_(self):
        return self.gate, self._group_interchangeable_qubits()

    def _decompose_(self) -> op_tree.OP_TREE:
        """List of gate operations that correspond to applying the single qubit
           gate to each of the target qubits individually
        """
        return [self.gate.on(qubit) for qubit in self.qubits]

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, None, NotImplementedType]:
        """Replicates the logic the simulators use to apply the equivalent
           sequence of GateOperations
        """
        state = args.target_tensor
        for axis in args.axes:
            result = protocols.apply_unitary(self.gate,
                                             protocols.ApplyUnitaryArgs(
                                                 state,
                                                 args.available_buffer,
                                                 (axis,)),
                                             default=NotImplemented)

            if result is args.available_buffer:
                args.available_buffer = state

            state = result

        return state

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        # Obtain the unitary for the single qubit gate
        single_unitary = protocols.unitary(self.gate, NotImplemented)

        # Make sure we actually have a matrix
        if single_unitary is NotImplemented:
            return single_unitary

        # Create a unitary which corresponds to applying the single qubit
        # unitary to each qubit. This will blow up memory fast.
        unitary = single_unitary
        for _ in range(len(self.qubits) - 1):
            unitary = np.outer(unitary, single_unitary)

        return unitary

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.gate)

    def _resolve_parameters_(self, resolver):
        resolved_gate = protocols.resolve_parameters(self.gate, resolver)
        return self.with_gate(resolved_gate)

    def _formatted_exponent(self, info: protocols.CircuitDiagramInfo,
                            args: protocols.CircuitDiagramInfoArgs
                            ) -> Optional[str]:
        if info.exponent == 0:
            return '0'

        # 1 is not shown.
        if info.exponent == 1:
            return None

        # Round -1.0 into -1.
        if info.exponent == -1:
            return '-1'

        # If it's a float, show the desired precision.
        if isinstance(info.exponent, float):
            if args.precision is not None:
                # funky behavior of fraction, cast to str in constructor helps.
                approx_frac = Fraction(info.exponent).limit_denominator(16)
                if approx_frac.denominator not in [2, 4, 5, 10]:
                    if abs(float(approx_frac)
                           - info.exponent) < 10 ** -args.precision:
                        return '({})'.format(approx_frac)

                return '{{:.{}}}'.format(args.precision).format(info.exponent)
        return repr(info.exponent)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        diagram_info = protocols.circuit_diagram_info(self.gate,
                                                      args,
                                                      NotImplemented)
        if diagram_info == NotImplemented:
            return diagram_info

        # Extract the formatted exponent and place it in wire symbols so that
        # every gate on the diagram has an exponent instead of the last one
        symbol = diagram_info.wire_symbols[0]
        exponent = self._formatted_exponent(diagram_info, args)
        if exponent is not None:
            symbol = "{0}^{1}".format(symbol, exponent)

        # Include every gate on the diagram instead of one
        wire_symbols = (symbol,) * len(self.qubits)
        # Set exponent=1 so that another extra exponent doesn't get tacked on
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols,
                                            exponent=1,
                                            connected=False)

    def _phase_by_(self, phase_turns: float,
                   qubit_index: int) -> 'ParallelGateOperation':
        phased_gate = protocols.phase_by(self._gate, phase_turns, qubit_index,
                                         default=None)
        if phased_gate is None:
            return NotImplemented
        return self.with_gate(phased_gate)

    def _trace_distance_bound_(self) -> float:
        return 1.

    def __pow__(self, exponent: Any) -> 'ParallelGateOperation':
        """Raise gate to a power, then reapply to the same qubits.

        Only works if the gate implements cirq.ExtrapolatableEffect.
        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5)(qubit)  or  G(qubit) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            A new operation on the same qubits with the scaled gate.
        """
        new_gate = protocols.pow(self.gate,
                                 exponent,
                                 NotImplemented)
        if new_gate is NotImplemented:
            return NotImplemented
        return self.with_gate(new_gate)

    def _qasm_(self, args: protocols.QasmArgs) -> Optional[str]:
        """
        Strings together the qasm for the gate acting on each qubit individually
        """
        qasm_list = [protocols.qasm(self.gate,
                                    args=args,
                                    qubits=[qubit],
                                    default=None) for qubit in self.qubits]
        return ''.join([qasm for qasm in qasm_list if qasm is not None])
