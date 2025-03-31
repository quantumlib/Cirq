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


from copy import copy, deepcopy
from typing import Any, Dict, Generic, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np

from cirq import ops, qis, sim
from cirq.sim.simulation_state import SimulationState, TSimulationState
from cirq.value import big_endian_int_to_bits

if TYPE_CHECKING:
    import cirq


def _is_identity(action) -> bool:
    """Check if the given action is equivalent to an identity."""
    gate = action.gate if isinstance(action, ops.Operation) else action
    if isinstance(gate, (ops.XPowGate, ops.CXPowGate, ops.CCXPowGate, ops.SwapPowGate)):
        return gate.exponent % 2 == 0
    return False


class ClassicalBasisState(qis.QuantumStateRepresentation):
    """Represents a classical basis state for efficient state evolution."""

    def __init__(self, initial_state: Union[List[int], np.ndarray]):
        """Initializes the ClassicalBasisState object.

        Args:
            initial_state: The initial state in the computational basis.
        """
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ClassicalBasisState':
        """Creates a copy of the ClassicalBasisState object.

        Args:
            deep_copy_buffers: Whether to deep copy the internal buffers.
        Returns:
            A copy of the ClassicalBasisState object.
        """
        return ClassicalBasisState(
            initial_state=deepcopy(self.basis) if deep_copy_buffers else copy(self.basis)
        )

    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        """Measures the density matrix.

        Args:
            axes: The axes to measure.
            seed: The random number seed to use.
        Returns:
            The measurements in order.
        """
        return [self.basis[i] for i in axes]


class ClassicalBasisSimState(SimulationState[ClassicalBasisState]):
    """Represents the state of a quantum simulation using classical basis states."""

    def __init__(
        self,
        initial_state: Union[int, List[int]] = 0,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Initializes the ClassicalBasisSimState object.

        Args:
            qubits: The qubits to simulate.
            initial_state: The initial state for the simulation.
            classical_data: The classical data container for the simulation.

        Raises:
            ValueError: If qubits not provided and initial_state is int.
                        If initial_state is not an int, List[int], or np.ndarray.

        An initial_state value of type integer is parsed in big endian order.
        """
        if isinstance(initial_state, int):
            if qubits is None:
                raise ValueError('qubits must be provided if initial_state is not List[int]')
            state = ClassicalBasisState(
                big_endian_int_to_bits(initial_state, bit_count=len(qubits))
            )
        elif isinstance(initial_state, (list, np.ndarray)):
            state = ClassicalBasisState(initial_state)
        else:
            raise ValueError('initial_state must be an int or List[int] or np.ndarray')
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True):
        """Acts on the state with a given operation.

        Args:
            action: The operation to apply.
            qubits: The qubits to apply the operation to.
            allow_decompose: Whether to allow decomposition of the operation.

        Returns:
            True if the operation was applied successfully.

        Raises:
            ValueError: If initial_state shape for type np.ndarray is not equal to 1.
                        If gate is not one of X, SWAP, a controlled version of X or SWAP,
                            or a measurement.
        """
        if isinstance(self._state.basis, np.ndarray) and len(self._state.basis.shape) != 1:
            raise ValueError('initial_state shape for type np.ndarray is not equal to 1')
        gate = action.gate if isinstance(action, ops.Operation) else action
        mapped_qubits = [self.qubit_map[i] for i in qubits]

        if isinstance(gate, ops.ControlledGate):
            control_qubits = mapped_qubits[: gate.num_controls()]
            mapped_qubits = mapped_qubits[gate.num_controls() :]

            controls_state = tuple(self._state.basis[c] for c in control_qubits)
            if controls_state not in gate.control_values.expand():
                # gate has no effect; controls were off
                return True

            gate = gate.sub_gate

        if _is_identity(gate):
            pass
        elif gate == ops.X:
            (q,) = mapped_qubits
            self._state.basis[q] ^= 1
        elif gate == ops.CNOT:
            c, q = mapped_qubits
            self._state.basis[q] ^= self._state.basis[c]
        elif gate == ops.SWAP:
            a, b = mapped_qubits
            self._state.basis[a], self._state.basis[b] = self._state.basis[b], self._state.basis[a]
        elif gate == ops.TOFFOLI:
            c1, c2, q = mapped_qubits
            self._state.basis[q] ^= self._state.basis[c1] & self._state.basis[c2]
        else:
            raise ValueError(
                f'{gate} is not one of X, SWAP; a controlled version '
                'of X or SWAP; or a measurement'
            )
        return True


class ClassicalStateStepResult(
    sim.StepResultBase['ClassicalBasisSimState'], Generic[TSimulationState]
):
    """The step result provided by `ClassicalStateSimulator.simulate_moment_steps`."""


class ClassicalStateTrialResult(
    sim.SimulationTrialResultBase['ClassicalBasisSimState'], Generic[TSimulationState]
):
    """The trial result provided by `ClassicalStateSimulator.simulate`."""


class ClassicalStateSimulator(
    sim.SimulatorBase[
        ClassicalStateStepResult['ClassicalBasisSimState'],
        ClassicalStateTrialResult['ClassicalBasisSimState'],
        'ClassicalBasisSimState',
    ],
    Generic[TSimulationState],
):
    """A simulator that accepts only gates with classical counterparts."""

    def __init__(
        self, *, noise: 'cirq.NOISE_MODEL_LIKE' = None, split_untangled_states: bool = False
    ):
        """Initializes a ClassicalStateSimulator.

        Args:
            noise: The noise model used by the simulator.
            split_untangled_states: Whether to run the simulation as a product state.

        Raises:
            ValueError: If noise_model is not None.
        """
        if noise is not None:
            raise ValueError(f'{noise=} is not supported')
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[ClassicalBasisSimState]',
    ) -> 'ClassicalStateTrialResult[ClassicalBasisSimState]':
        """Creates a trial result for the simulator.

        Args:
            params: The parameter resolver for the simulation.
            measurements: The measurement results.
            final_simulator_state: The final state of the simulator.
        Returns:
            A trial result for the simulator.
        """
        return ClassicalStateTrialResult(
            params, measurements, final_simulator_state=final_simulator_state
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[ClassicalBasisSimState]'
    ) -> 'ClassicalStateStepResult[ClassicalBasisSimState]':
        """Creates a step result for the simulator.

        Args:
            sim_state: The current state of the simulator.
        Returns:
            A step result for the simulator.
        """
        return ClassicalStateStepResult(sim_state)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> 'ClassicalBasisSimState':
        """Creates a partial simulation state for the simulator.

        Args:
            initial_state: The initial state for the simulation.
            qubits: The qubits associated with the state.
            classical_data: The shared classical data container for this simulation.
        Returns:
            A partial simulation state.
        """
        return ClassicalBasisSimState(
            initial_state=initial_state, qubits=qubits, classical_data=classical_data
        )
