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

from cirq import ops
from cirq.ops.raw_types import Qid
from typing import List, Sequence, Union, Optional, Tuple
from cirq.sim.simulation_state import SimulationState
from cirq import qis
from cirq.value import big_endian_int_to_bits
import numpy as np



from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING

import numpy as np

from cirq import sim
from cirq.sim.simulation_state import TSimulationState

if TYPE_CHECKING:
    import cirq

def _is_identity(gate: ops.GateOperation) -> bool:
    if isinstance(gate, (ops.XPowGate, ops.CXPowGate, ops.CCXPowGate, ops.SwapPowGate)):
        return gate.exponent % 2 == 0
    return False


class ComputationalBasisState(qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ComputationalBasisState':
        return ComputationalBasisState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]


class ComputationalBasisSimState(SimulationState[ComputationalBasisState]):
    """A simulator that accepts only gates with classical counterparts.
    This simulator evolves a single state, using only gates that output a single state for each
    input state. The simulator runs in linear time, at the cost of not supporting superposition.
    It can be used to estimate costs and simulate circuits for simple non-quantum algorithms using
    many more qubits than fully capable quantum simulators.

    The supported gates are:
        - cirq.X
        - cirq.CNOT
        - cirq.SWAP
        - cirq.TOFFOLI
        - cirq.measure

    Args:
        circuit: The circuit to simulate.
        param_resolver: Parameters to run with the program.
        repetitions: Number of times to repeat the run. It is expected that
            this is validated greater than zero before calling this method.

    Returns:
        A dictionary mapping measurement keys to measurement results.

    Raises:
        ValueError: If
            - one of the gates is not an X, CNOT, SWAP, TOFFOLI or a measurement.
            - A measurement key is used for measurements on different numbers of qubits.
    """
    def __init__(
        self, 
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0, 
        qubits: Optional[Sequence['cirq.Qid']] = [], 
        classical_data: Optional['cirq.ClassicalDataStore'] = None
    ):
        state = ComputationalBasisState(big_endian_int_to_bits(initial_state, bit_count=len(qubits)))
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, ops.Operation) else action
        mapped_qubits = [self.qubit_map[i] for i in qubits]
        if _is_identity(gate):
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
            raise ValueError(
                f'{gate} is not one of cirq.X, cirq.CNOT, cirq.SWAP, '
                'cirq.CCNOT, or a measurement'
            )

            
class ClassicalStateStepResult(sim.StepResultBase[TSimulationState], Generic[TSimulationState]):
    """The step result provided by `ClassicalStateSimulator.simulate_moment_steps`."""


class ClassicalStateTrialResult(
    sim.SimulationTrialResultBase[TSimulationState], Generic[TSimulationState]
):
    """The trial result provided by `ClassicalStateSimulator.simulate`."""


class ClassicalStateSimulator(
    sim.SimulatorBase[
        ClassicalStateStepResult[TSimulationState],
        ClassicalStateTrialResult[TSimulationState],
        TSimulationState,
    ],
    Generic[TSimulationState],
):
    """A simulator that can be used to simulate classical states."""

    def __init__(
        self,
        state_type: Type[TSimulationState] = ComputationalBasisSimState,
        *,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        split_untangled_states: bool = False,
    ):
        """Initializes a ClassicalStateSimulator.

        Args:
            state_type: The class that represents the simulation state this simulator should use.
            noise: The noise model used by the simulator.
            split_untangled_states: True to run the simulation as a product state. This is only
                supported if the `state_type` supports it via an implementation of `kron` and
                `factor` methods. Otherwise a runtime error will occur during simulation."""
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)
        self.state_type = state_type

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[TSimulationState]',
    ) -> 'ClassicalStateTrialResult[TSimulationState]':
        return ClassicalStateTrialResult(
            params, measurements, final_simulator_state=final_simulator_state
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[TSimulationState]'
    ) -> 'ClassicalStateStepResult[TSimulationState]':
        return ClassicalStateStepResult(sim_state)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> TSimulationState:
        return self.state_type(
            initial_state=initial_state, qubits=qubits, classical_data=classical_data
        )  # type: ignore[call-arg]



