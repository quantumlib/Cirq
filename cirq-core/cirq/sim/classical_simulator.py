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


from typing import Dict, Generic, Any, Sequence, TYPE_CHECKING, List
from cirq import ops, qis, value
from cirq.value import big_endian_int_to_bits
from cirq.ops.raw_types import Qid
from cirq import sim
from cirq.sim.simulation_state import SimulationState, SimulationStateBase
import numpy as np

if TYPE_CHECKING:
    import cirq


class ClassicalBasisState(qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ClassicalBasisState':
        basis_copy = self.basis if deep_copy_buffers else self.basis.copy()
        return ClassicalBasisState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]

    def sample(
        self,
        axes: Sequence[int],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples the ClassicalBasisState.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """

        measurements: List[List[int]] = []
        prng = value.parse_random_state(seed) if seed else self._prng

        for _ in range(repetitions):
            measurements.append(self.measure(axes, prng))

        return np.array(measurements, dtype=int)

    def create_merged_state(self) -> 'ClassicalBasisSimState':
        final_args = self.sim_state[None]
        for args in set([self.sim_state[k] for k in self.sim_state.keys() if k is not None]):
            final_args = final_args.kronecker_product(args)
        return final_args.transpose_to_qubit_order(self.basis)


class ClassicalBasisSimState(SimulationState[ClassicalBasisState]):
    def __init__(self, initial_state, qubits, classical_data):
        state = ClassicalBasisState(big_endian_int_to_bits(initial_state, bit_count=len(qubits)))
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, ops.Operation) else action
        mapped_qubits = [self.qubit_map[i] for i in qubits]
        if self._is_identity(gate):
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
            raise ValueError(f'{gate} is not one of X, CNOT, SWAP, CCNOT, or a measurement')

    def _is_identity(self, action) -> bool:
        gate = action.gate if isinstance(action, ops.Operation) else action
        if isinstance(gate, (ops.XPowGate, ops.CXPowGate, ops.CCXPowGate, ops.SwapPowGate)):
            return gate.exponent % 2 == 0
        return False


class ClassicalStateStepResult(sim.StepResultBase['ClassicalBasisSimState'], Generic['ClassicalBasisSimState']):
    """The step result provided by `ClassicalStateSimulator.simulate_moment_steps`."""

    def __init__(self, sim_state: SimulationStateBase['ClassicalBasisSimState']):
        """Initializes the step result.

        Args:
            sim_state: The `SimulationStateBase` for this step.
        """
        super().__init__(sim_state)

    @property
    def state(self):
        return self._merged_sim_state


class ClassicalStateTrialResult(
    sim.SimulationTrialResultBase['ClassicalBasisSimState'], Generic['ClassicalBasisSimState']
):
    """The trial result provided by `ClassicalStateSimulator.simulate`."""

    def __init__(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[ClassicalBasisSimState]',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )


class ClassicalStateSimulator(
    sim.SimulatorBase[
        ClassicalStateStepResult['ClassicalBasisSimState'],
        ClassicalStateTrialResult['ClassicalBasisSimState'],
        'ClassicalBasisSimState',
    ],
    Generic['ClassicalBasisSimState'],
):
    """A simulator that accepts only gates with classical counterparts."""

    def __init__(
        self, *, noise: 'cirq.NOISE_MODEL_LIKE' = None, split_untangled_states: bool = False
    ):
        """Initializes a CustomStateSimulator.

        Args:
            state_type: The class that represents the simulation state this simulator should use.
            noise: The noise model used by the simulator.
            split_untangled_states: True to run the simulation as a product state. This is only
                supported if the `state_type` supports it via an implementation of `kron` and
                `factor` methods. Otherwise a runtime error will occur during simulation."""
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[ClassicalBasisSimState]',
    ) -> 'ClassicalStateTrialResult[ClassicalBasisSimState]':
        return ClassicalStateTrialResult(
            params, measurements, final_simulator_state=final_simulator_state
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[ClassicalBasisSimState]'
    ) -> 'ClassicalStateStepResult[ClassicalBasisSimState]':
        return ClassicalStateStepResult(sim_state)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> 'ClassicalBasisSimState':
        if isinstance(initial_state, ClassicalBasisSimState):
            return initial_state
        return ClassicalBasisSimState(
            initial_state=initial_state, qubits=qubits, classical_data=classical_data
        )
