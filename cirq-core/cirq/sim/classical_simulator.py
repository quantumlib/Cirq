# Copyright 2023 The Cirq Developers
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

#both interfaces
from collections import defaultdict
from cirq import ops, protocols
from cirq.study.resolver import ParamResolver
from cirq.circuits.circuit import AbstractCircuit
from cirq.ops.raw_types import Qid

# #SimulatesSamples interfaces
# from typing import Dict
# from cirq.sim.simulator import SimulatesSamples
# import numpy as np

#SimulationState interfaces
from typing import List, Sequence
from cirq.sim.simulation_state import SimulationState
from cirq import qis
from cirq.value import big_endian_int_to_bits


def _is_identity(gate: ops.GateOperation) -> bool:
    if isinstance(gate, (ops.XPowGate, ops.CXPowGate, ops.CCXPowGate, ops.SwapPowGate)):
        return gate.exponent % 2 == 0
    return False


# class ClassicalStateSimulator(SimulatesSamples):
#     """A simulator that accepts only gates with classical counterparts.

#     This simulator evolves a single state, using only gates that output a single state for each
#     input state. The simulator runs in linear time, at the cost of not supporting superposition.
#     It can be used to estimate costs and simulate circuits for simple non-quantum algorithms using
#     many more qubits than fully capable quantum simulators.

#     The supported gates are:
#         - cirq.X
#         - cirq.CNOT
#         - cirq.SWAP
#         - cirq.TOFFOLI
#         - cirq.measure

#     Args:
#         circuit: The circuit to simulate.
#         param_resolver: Parameters to run with the program.
#         repetitions: Number of times to repeat the run. It is expected that
#             this is validated greater than zero before calling this method.

#     Returns:
#         A dictionary mapping measurement keys to measurement results.

#     Raises:
#         ValueError: If
#             - one of the gates is not an X, CNOT, SWAP, TOFFOLI or a measurement.
#             - A measurement key is used for measurements on different numbers of qubits.
#     """

#     def _run(
#         self, circuit: AbstractCircuit, param_resolver: ParamResolver, repetitions: int
#     ) -> Dict[str, np.ndarray]:
#         results_dict: Dict[str, np.ndarray] = {}
#         values_dict: Dict[Qid, int] = defaultdict(int)
#         param_resolver = param_resolver or ParamResolver({})
#         resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)

#         for moment in resolved_circuit:
#             for op in moment:
#                 if _is_identity(op):
#                     continue
#                 if op.gate == ops.X:
#                     (q,) = op.qubits
#                     values_dict[q] ^= 1
#                 elif op.gate == ops.CNOT:
#                     c, q = op.qubits
#                     values_dict[q] ^= values_dict[c]
#                 elif op.gate == ops.SWAP:
#                     a, b = op.qubits
#                     values_dict[a], values_dict[b] = values_dict[b], values_dict[a]
#                 elif op.gate == ops.TOFFOLI:
#                     c1, c2, q = op.qubits
#                     values_dict[q] ^= values_dict[c1] & values_dict[c2]
#                 elif protocols.is_measurement(op):
#                     measurement_values = np.array(
#                         [[[values_dict[q] for q in op.qubits]]] * repetitions, dtype=np.uint8
#                     )
#                     key = op.gate.key  # type: ignore
#                     if key in results_dict:
#                         if op._num_qubits_() != results_dict[key].shape[-1]:
#                             raise ValueError(
#                                 f'Measurement shape {len(measurement_values)} does not match '
#                                 f'{results_dict[key].shape[-1]} in {key}.'
#                             )
#                         results_dict[key] = np.concatenate(
#                             (results_dict[key], measurement_values), axis=1
#                         )
#                     else:
#                         results_dict[key] = measurement_values
#                 else:
#                     raise ValueError(
#                         f'{op} is not one of cirq.X, cirq.CNOT, cirq.SWAP, '
#                         'cirq.CCNOT, or a measurement'
#                     )

#         return results_dict

class ClassicalState(qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ComputationalBasisState':
        return ClassicalState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]


class ClassicalStateSimulator(SimulationState[ClassicalState]):
    def __init__(self, initial_state, qubits, classical_data):
        state = ClassicalState(big_endian_int_to_bits(initial_state, bit_count=len(qubits)))
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, ops.Operation) else action
        mapped_qubits = [self.qubit_map[i] for i in qubits]
        if _is_identity(action):
            return True
        if gate == ops.X:
            (q,) = mapped_qubits
            self._state.basis[q] ^= 1
            return True
        elif gate == ops.CNOT:
            c, q = mapped_qubits
            self._state.basis[q] ^= self._state.basis[c]
            return True
        elif gate == ops.SWAP:
            a, b = mapped_qubits
            self._state.basis[a], self._state.basis[b] = self._state.basis[b], self._state.basis[a]
            return True
        elif gate == ops.TOFFOLI:
            c1, c2, q = mapped_qubits
            self._state.basis[q] ^= self._state.basis[c1] & self._state.basis[c2]
            return True

        else:
            print(f'Unsupported operation: {gate!r}')
            return False
