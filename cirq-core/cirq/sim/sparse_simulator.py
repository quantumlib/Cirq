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

from typing import Any, Iterator, List, Optional, Sequence, Type, TYPE_CHECKING, Union

import numpy as np

from cirq import ops
from cirq.sim import simulator, state_vector, state_vector_simulation_state, state_vector_simulator

if TYPE_CHECKING:
    import cirq


class Simulator(
    state_vector_simulator.SimulatesIntermediateStateVector['SparseSimulatorStep'],
    simulator.SimulatesExpectationValues,
):
    """A sparse matrix state vector simulator that uses numpy.

    This simulator can be applied on circuits that are made up of operations
    that have a `_unitary_` method, or `_has_unitary_` and
    `_apply_unitary_`, `_mixture_` methods, are measurements, or support a
    `_decompose_` method that returns operations satisfying these same
    conditions. That is to say, the operations should follow the
    `cirq.SupportsConsistentApplyUnitary` protocol, the `cirq.SupportsUnitary`
    protocol, the `cirq.SupportsMixture` protocol, or the
    `cirq.SupportsDecompose` protocol. It is also permitted for the circuit
    to contain measurements which are operations that support
    `cirq.SupportsKraus` and `cirq.SupportsMeasurementKey`

    This can run simulations which mimic use of actual quantum hardware.
    These simulations do not give access to the state vector (like actual
    hardware).  There are two variations of run methods, one which takes in a
    single (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolvers:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    The simulation performs optimizations if the number of repetitions is
    greater than one and all measurements in the circuit are terminal (at the
    end of the circuit). These methods return `Result`s which contain both
    the measurement results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast, the simulate methods of the simulator give access to the
    state vector of the simulation at the end of the simulation of the circuit.
    These methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `DensityMatrixTrialResult` for details of this ordering). The initial
    state can be either the full state vector, or an integer which represents
    the initial state of being in a computational basis state for the binary
    representation of that integer. Similar to run methods, there are two
    simulate methods that run for single runs or for sweeps across different
    parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods, in contrast to the run methods, do not perform
    repetitions. The result of these simulations is a
    `SimulationTrialResult` which contains measurement
    results, information about parameters used in the simulation, and
    access to the state via the `state` method and
    `cirq.sim.state_vector.StateVectorMixin` methods.

    If one wishes to perform simulations that have access to the
    state vector as one steps through running the circuit, there is a generator
    which can be iterated over.  Each step is an object that gives access
    to the state vector.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments with the following (replace 'sim'
    with your `Simulator` object):

        for step_result in sim.simulate_moment_steps(circuit):
           # do something with the state vector via step_result.state_vector

    Note also that simulations can be stochastic, i.e. return different results
    for different runs.  The first version of this occurs for measurements,
    where the results of the measurement are recorded.  This can also
    occur when the circuit has mixtures of unitaries.

    If only the expectation values for some observables on the final state are
    required, there are methods for that as well. These methods take a mapping
    of names to observables, and return a map (or list of maps) of those names
    to the corresponding expectation values.

        simulate_expectation_values(circuit, observables, param_resolver,
                                    qubit_order, initial_state,
                                    permit_terminal_measurements)

        simulate_expectation_values_sweep(circuit, observables, params,
                                          qubit_order, initial_state,
                                          permit_terminal_measurements)

    Expectation values generated by these methods are exact (up to precision of
    the floating-point type used); the closest analogy on hardware requires
    estimating the expectation values from several samples.

    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(
        self,
        *,
        dtype: Type[np.complexfloating] = np.complex64,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        split_untangled_states: bool = True,
    ):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.

        Raises:
            ValueError: If the given dtype is not complex.
        """
        if np.dtype(dtype).kind != 'c':
            raise ValueError(f'dtype must be a complex type but was {dtype}')
        super().__init__(
            dtype=dtype, noise=noise, seed=seed, split_untangled_states=split_untangled_states
        )

    def _create_partial_simulation_state(
        self,
        initial_state: Union['cirq.STATE_VECTOR_LIKE', 'cirq.StateVectorSimulationState'],
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ):
        """Creates the StateVectorSimulationState for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.

        Returns:
            StateVectorSimulationState for the circuit.
        """
        if isinstance(initial_state, state_vector_simulation_state.StateVectorSimulationState):
            # Instances of SimulationStateBase usually returned before this point
            return initial_state  # pragma: no cover

        return state_vector_simulation_state.StateVectorSimulationState(
            qubits=qubits,
            prng=self._prng,
            classical_data=classical_data,
            initial_state=initial_state,
            dtype=self._dtype,
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]'
    ):
        return SparseSimulatorStep(sim_state=sim_state, dtype=self._dtype)

    def simulate_expectation_values_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> Iterator[List[float]]:
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError(
                'Provided circuit has terminal measurements, which may '
                'skew expectation values. If this is intentional, set '
                'permit_terminal_measurements=True.'
            )
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qmap = {q: i for i, q in enumerate(qubit_order.order_for(program.all_qubits()))}
        if not isinstance(observables, List):
            observables = [observables]
        pslist = [ops.PauliSum.wrap(pslike) for pslike in observables]
        yield from (
            [obs.expectation_from_state_vector(result.final_state_vector, qmap) for obs in pslist]
            for result in self.simulate_sweep_iter(
                program, params, qubit_order=qubit_order, initial_state=initial_state
            )
        )


class SparseSimulatorStep(
    state_vector.StateVectorMixin, state_vector_simulator.StateVectorStepResult
):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(
        self,
        sim_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]',
        dtype: Type[np.complexfloating] = np.complex64,
    ):
        """Results of a step of the simulator.

        Args:
            sim_state: The qubit:SimulationState lookup for this step.
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
        """
        qubit_map = {q: i for i, q in enumerate(sim_state.qubits)}
        super().__init__(sim_state=sim_state, qubit_map=qubit_map)
        self._dtype = dtype
        self._state_vector: Optional[np.ndarray] = None

    def state_vector(self, copy: bool = False):
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
        if self._state_vector is None:
            self._state_vector = np.array([1])
            state = self._merged_sim_state
            if state is not None:
                vector = state.target_tensor
                size = np.prod(vector.shape, dtype=np.int64)
                self._state_vector = np.reshape(vector, size)
        return self._state_vector.copy() if copy else self._state_vector

    def __repr__(self) -> str:
        # Dtype doesn't have a good repr, so we work around by invoking __name__.
        return (
            f'cirq.SparseSimulatorStep(sim_state={self._sim_state!r},'
            f' dtype=np.{np.dtype(self._dtype)!r})'
        )
