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

"""A simulator that uses numpy's einsum for sparse matrix operations."""
from typing import (
    Dict,
    List,
    Type,
    TYPE_CHECKING,
    Any,
)

import numpy as np

from cirq import ops, protocols, qis, study, value
from cirq.sim import (
    simulator,
    state_vector,
    state_vector_simulator,
)
from cirq.sim.act_on_state_vector_args import ActOnStateVectorArgs
from cirq.sim.op_by_op_simulator import (
    StateFactory,
    SimulationResultFactory,
    OpByOpSimulator,
)

from cirq.sim.state_vector_simulator import StateVectorTrialResult, StateVectorSimulatorState

if TYPE_CHECKING:
    import cirq


class SparseStateFactory(StateFactory[ActOnStateVectorArgs]):
    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ):
        if np.dtype(dtype).kind != 'c':
            raise ValueError('dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)

    def create_sim_state(self, initial_state, qubits):
        num_qubits = len(qubits)
        qid_shape = protocols.qid_shape(qubits)
        state = qis.to_valid_state_vector(
            initial_state, num_qubits, qid_shape=qid_shape, dtype=self._dtype
        )
        sim_state = ActOnStateVectorArgs(
            target_tensor=np.reshape(state, qid_shape),
            available_buffer=np.empty(qid_shape, dtype=self._dtype),
            axes=[],
            prng=self._prng,
            log_of_measurement_results={},
        )
        return sim_state

    def act_on_state(self, op, sim_state, qubit_map):
        sim_state.axes = tuple(qubit_map[qubit] for qubit in op.qubits)
        protocols.act_on(op, sim_state)

    @property
    def prng(self):
        return self._prng


class SparseSimulatorStep(
    state_vector.StateVectorMixin, state_vector_simulator.StateVectorStepResult
):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state_vector, measurements, qubit_map, dtype):
        """Results of a step of the simulator.

        Args:
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
        """
        super().__init__(measurements=measurements, qubit_map=qubit_map)
        self._dtype = dtype
        size = np.prod(protocols.qid_shape(self), dtype=int)
        self._state_vector = np.reshape(state_vector, size)

    def _simulator_state(self) -> state_vector_simulator.StateVectorSimulatorState:
        return state_vector_simulator.StateVectorSimulatorState(
            qubit_map=self.qubit_map, state_vector=self._state_vector
        )

    def state_vector(self, copy: bool = True):
        """Return the state vector at this point in the computation.

        The state is returned in the computational basis with these basis
        states defined by the qubit_map. In particular the value in the
        qubit_map is the index of the qubit, and these are translated into
        binary vectors where the last qubit is the 1s bit of the index, the
        second-to-last is the 2s bit of the index, and so forth (i.e. big
        endian ordering).

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned vector will have indices mapped to qubit basis
             states like the following table

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |

        Args:
            copy: If True, then the returned state is a copy of the state
                vector. If False, then the state vector is not copied,
                potentially saving memory. If one only needs to read derived
                parameters from the state vector and store then using False
                can speed up simulation by eliminating a memory copy.
        """
        vector = self._simulator_state().state_vector
        return vector.copy() if copy else vector

    def set_state_vector(self, state: 'cirq.STATE_VECTOR_LIKE'):
        update_state = qis.to_valid_state_vector(
            state, len(self.qubit_map), qid_shape=protocols.qid_shape(self, None), dtype=self._dtype
        )
        np.copyto(self._state_vector, update_state)

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self.qubit_map[qubit] for qubit in qubits]
        return state_vector.sample_state_vector(
            self._state_vector,
            indices,
            qid_shape=protocols.qid_shape(self, None),
            repetitions=repetitions,
            seed=seed,
        )


class SparseSimulationResultFactory(
    SimulationResultFactory[
        ActOnStateVectorArgs, SparseSimulatorStep, StateVectorTrialResult, StateVectorSimulatorState
    ]
):
    def trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any,
    ) -> StateVectorTrialResult:
        return StateVectorTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def step_result(self, sim_state, qubit_map):
        return SparseSimulatorStep(
            state_vector=sim_state.target_tensor,
            measurements=dict(sim_state.log_of_measurement_results),
            qubit_map=qubit_map,
            dtype=sim_state.dtype,
        )


class Simulator(
    OpByOpSimulator[
        ActOnStateVectorArgs, SparseSimulatorStep, StateVectorTrialResult, StateVectorSimulatorState
    ],
    simulator.SimulatesSamples,
    state_vector_simulator.SimulatesIntermediateStateVector,
):
    """A sparse matrix state vector simulator that uses numpy.

    This simulator can be applied on circuits that are made up of operations
    that have a `_unitary_` method, or `_has_unitary_` and
    `_apply_unitary_`, `_mixture_` methods, are measurements, or support a
    `_decompose_` method that returns operations satisfying these same
    conditions. That is to say, the operations should follow the
    `cirq.SupportsConsistentApplyUnitary` protocol, the `cirq.SupportsUnitary`
    protocol, the `cirq.SupportsMixture` protocol, or the
    `cirq.CompositeOperation` protocol. It is also permitted for the circuit
    to contain measurements which are operations that support
    `cirq.SupportsChannel` and `cirq.SupportsMeasurementKey`

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the state vector (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    The simulation performs optimizations if the number of repetitions is
    greater than one and all measurements in the circuit are terminal (at the
    end of the circuit). These methods return `Result`s which contain both
    the measurement results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the
    state vector of the simulation at the end of the simulation of the circuit.
    These methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `SparseSimulationTrialResult` for details of this ordering). The initial
    state can be either the full state vector, or an integer which represents
    the initial state of being in a computational basis state for the binary
    representation of that integer. Similar to run methods, there are two
    simulate methods that run for single runs or for sweeps across different
    parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `SparseSimulationTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation,access to the state via the `state` method and `StateVectorMixin`
    methods.

    If one wishes to perform simulations that have access to the
    state vector as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the state vector.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the state vector via step_result.state_vector

    Note also that simulations can be stochastic, i.e. return different results
    for different runs.  The first version of this occurs for measurements,
    where the results of the measurement are recorded.  This can also
    occur when the circuit has mixtures of unitaries.

    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
            seed: The random seed to use for this simulator.
        """
        state_algo = SparseStateFactory(dtype=dtype, seed=seed)
        result_producer = SparseSimulationResultFactory()
        super().__init__(state_algo, result_producer)
