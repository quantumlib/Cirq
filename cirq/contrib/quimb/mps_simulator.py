# Copyright 2019 The Cirq Developers
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
"""An MPS simulator.

This is based on this paper:
https://arxiv.org/abs/2002.07730
"""

import collections
import math
from typing import Any, Dict, List, Iterator, Optional, Sequence, Set

import dataclasses
import numpy as np
import quimb.tensor as qtn

import cirq
from cirq import circuits, study, ops, protocols, value
from cirq.sim import simulator


@dataclasses.dataclass(frozen=True)
class MPSOptions:
    # Some of these parameters are fed directly to Quimb so refer to the documentation for detail:
    # https://quimb.readthedocs.io/en/latest/_autosummary/ \
    #       quimb.tensor.tensor_core.html#quimb.tensor.tensor_core.tensor_split

    # How to split the tensor. Refer to the Quimb documentation for the exact meaning.
    method: str = 'svds'
    # If integer, the maxmimum number of singular values to keep, regardless of ``cutoff``.
    max_bond: Optional[int] = None
    # Method with which to apply the cutoff threshold. Refer to the Quimb documentation.
    cutoff_mode: str = 'rsum2'
    # The threshold below which to discard singular values. Refer to the Quimb documentation.
    cutoff: float = 1e-6
    # Because the computation is approximate, the sum of the probabilities is not 1.0. This
    # parameter is the absolute deviation from 1.0 that is allowed.
    sum_prob_atol: float = 1e-3


class MPSSimulator(
    simulator.SimulatesSamples,
    simulator.SimulatesIntermediateState[
        'cirq.contrib.quimb.mps_simulator.MPSSimulatorStepResult',
        'cirq.contrib.quimb.mps_simulator.MPSTrialResult',
        'cirq.contrib.quimb.mps_simulator.MPSState',
    ],
):
    """An efficient simulator for MPS circuits."""

    def __init__(
        self,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        simulation_options: 'cirq.contrib.quimb.mps_simulator.MPSOptions' = MPSOptions(),
        grouping: Optional[Dict['cirq.Qid', int]] = None,
    ):
        """Creates instance of `MPSSimulator`.

        Args:
            seed: The random seed to use for this simulator.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.
        """
        self.init = True
        self._prng = value.parse_random_state(seed)
        self.simulation_options = simulation_options
        self.grouping = grouping

    def _base_iterator(
        self, circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList, initial_state: int
    ) -> Iterator['cirq.contrib.quimb.mps_simulator.MPSSimulatorStepResult']:
        """Iterator over MPSSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.

        Yields:
            MPSStepResult from simulating a Moment of the Circuit.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())

        qubit_map = {q: i for i, q in enumerate(qubits)}

        if len(circuit) == 0:
            yield MPSSimulatorStepResult(
                measurements={},
                state=MPSState(
                    qubit_map,
                    self.simulation_options,
                    self.grouping,
                    initial_state=initial_state,
                ),
            )
            return

        state = MPSState(
            qubit_map,
            self.simulation_options,
            self.grouping,
            initial_state=initial_state,
        )

        for moment in circuit:
            measurements: Dict[str, List[int]] = collections.defaultdict(list)

            for op in moment:
                if isinstance(op.gate, ops.MeasurementGate):
                    key = str(protocols.measurement_key(op))
                    measurements[key].extend(state.perform_measurement(op.qubits, self._prng))
                elif protocols.has_unitary(op):
                    state.apply_unitary(op)
                else:
                    raise NotImplementedError(f"Unrecognized operation: {op!r}")

            yield MPSSimulatorStepResult(measurements=measurements, state=state)

    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: int,
    ) -> Iterator['cirq.contrib.quimb.mps_simulator.MPSSimulatorStepResult']:
        """Iterator over MPSSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            An interator over all the results.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state

        return self._base_iterator(resolved_circuit, qubit_order, actual_initial_state)

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.contrib.quimb.mps_simulator.MPSState',
    ) -> 'cirq.contrib.quimb.mps_simulator.MPSTrialResult':
        """Creates a single trial results with the measurements.

        Args:
            circuit: The circuit to simulate.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            measurements: A dictionary from measurement key (e.g. qubit) to the
                actual measurement array.
            final_simulator_state: The final state of the simulator.

        Returns:
            A single result.
        """
        return MPSTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, List[np.ndarray]]:
        """Repeats measurements multiple times.

        Args:
            circuit: The circuit to simulate.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            repetitions: How many measurements to perform
            final_simulator_state: The final state of the simulator.

        Returns:
            A dictionay of measurement key (e.g. qubit) to a list of arrays that
            are the measurements.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)

        measurements = {}  # type: Dict[str, List[np.ndarray]]
        if repetitions == 0:
            for _, op, _ in resolved_circuit.findall_operations_with_gate_type(ops.MeasurementGate):
                measurements[protocols.measurement_key(op)] = np.empty([0, 1])

        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                resolved_circuit, qubit_order=ops.QubitOrder.DEFAULT, initial_state=0
            )

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=int))

        return {k: np.array(v) for k, v in measurements.items()}

    def _check_all_resolved(self, circuit):
        """Raises if the circuit contains unresolved symbols."""
        if protocols.is_parameterized(circuit):
            unresolved = [
                op for moment in circuit for op in moment if protocols.is_parameterized(op)
            ]
            raise ValueError(
                'Circuit contains ops whose symbols were not specified in '
                'parameter sweep. Ops: {}'.format(unresolved)
            )


class MPSTrialResult(simulator.SimulationTrialResult):
    """A single trial reult"""

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'MPSState',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

        self.final_state = final_simulator_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._final_simulator_state
        return f'measurements: {samples}\noutput state: {final}'


class MPSSimulatorStepResult(simulator.StepResult['MPSState']):
    """A `StepResult` that can perform measurements."""

    def __init__(self, state, measurements):
        """Results of a step of the simulator.
        Attributes:
            state: A MPSState
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
        """
        self.measurements = measurements
        self.state = state.copy()

    def __str__(self) -> str:
        def bitstring(vals):
            return ','.join(str(v) for v in vals)

        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])

        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results]) + '\n'

        final = self.state

        return f'{measurements}{final}'

    def _simulator_state(self):
        return self.state

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:

        measurements: List[int] = []

        for _ in range(repetitions):
            measurements.append(
                self.state.perform_measurement(
                    qubits, value.parse_random_state(seed), collapse_state_vector=False
                )
            )

        return np.array(measurements, dtype=int)


@value.value_equality
class MPSState:
    """A state of the MPS simulation."""

    def __init__(
        self,
        qubit_map: Dict['cirq.Qid', int],
        simulation_options: 'cirq.contrib.quimb.mps_simulator.MPSOptions' = MPSOptions(),
        grouping: Optional[Dict['cirq.Qid', int]] = None,
        initial_state: int = 0,
    ):
        """Creates and MPSState

        Args:
            qubit_map: A map from Qid to an integer that uniquely identifies it.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.
            initial_state: An integer representing the initial state.
        """
        self.qubit_map = qubit_map
        self.grouping = qubit_map if grouping is None else grouping
        if self.grouping.keys() != self.qubit_map.keys():
            raise ValueError('Grouping must cover exactly the qubits.')
        self.M = []
        for _ in range(max(self.grouping.values()) + 1):
            self.M.append(qtn.Tensor())

        # The order of the qubits matters, because the state |01> is different from |10>. Since
        # Quimb uses strings to name tensor indices, we want to be able to sort them too. If we are
        # working with, say, 123 qubits then we want qubit 3 to come before qubit 100, but then
        # we want write the string '003' which comes before '100' in lexicographic order. The code
        # below is just simple string formatting.
        max_num_digits = len('{}'.format(max(qubit_map.values())))
        self.format_i = 'i_{{:0{}}}'.format(max_num_digits)
        self.format_mu = 'mu_{}_{}'

        # TODO(tonybruguier): Instead of relying on sortable indices could you keep a parallel
        # mapping of e.g. qubit to string-index and do all "logic" on the qubits themselves and
        # only translate to string-indices when calling a quimb API.

        # TODO(tonybruguier): Refactor out so that the code below can also be used by
        # circuit_to_tensors in cirq.contrib.quimb.state_vector.

        for qubit in reversed(list(qubit_map.keys())):
            d = qubit.dimension
            x = np.zeros(d)
            x[initial_state % d] = 1.0

            i = qubit_map[qubit]
            n = self.grouping[qubit]
            self.M[n] @= qtn.Tensor(x, inds=(self.i_str(i),))
            initial_state = initial_state // d
        self.simulation_options = simulation_options
        self.estimated_gate_error_list: List[float] = []

    def i_str(self, i: int) -> str:
        # Returns the index name for the i'th qid.
        return self.format_i.format(i)

    def mu_str(self, i: int, j: int) -> str:
        # Returns the index name for the pair of the i'th and j'th qids. Note
        # that by convention, the lower index is always the first in the output
        # string.
        smallest = min(i, j)
        largest = max(i, j)
        return self.format_mu.format(smallest, largest)

    def __str__(self) -> str:
        return str(qtn.TensorNetwork(self.M))

    def _value_equality_values_(self) -> Any:
        return self.qubit_map, self.M, self.simulation_options, self.grouping

    def copy(self) -> 'MPSState':
        state = MPSState(self.qubit_map, self.simulation_options, self.grouping)
        state.M = [x.copy() for x in self.M]
        state.estimated_gate_error_list = self.estimated_gate_error_list
        return state

    def state_vector(self) -> np.ndarray:
        """Returns the full state vector.

        Returns:
            A vector that contains the full state.
        """
        tensor_network = qtn.TensorNetwork(self.M)
        state_vector = tensor_network.contract(inplace=False)

        # Here, we rely on the formatting of the indices, and the fact that we have enough
        # leading zeros so that 003 comes before 100.
        sorted_ind = tuple(sorted(state_vector.inds))
        return state_vector.fuse({'i': sorted_ind}).data

    def partial_trace(self, keep_qubits: Set[ops.Qid]) -> np.ndarray:
        """Traces out all qubits except keep_qubits.

        Args:
            keep_qubits: The set of qubits that are left after computing the
                partial trace. For example, if we have a circuit for 3 qubits
                and this parameter only has one qubit, the entire density matrix
                would be 8x8, but this function returns a 2x2 matrix.

        Returns:
            An array that contains the partial trace.
        """

        contracted_inds = set(
            [self.i_str(i) for qubit, i in self.qubit_map.items() if qubit not in keep_qubits]
        )

        conj_pfx = "conj_"

        tensor_network = qtn.TensorNetwork(self.M)

        # Rename the internal indices to avoid collisions. Also rename the qubit
        # indices that are kept. We do not rename the qubit indices that are
        # traced out.
        conj_tensor_network = tensor_network.conj()
        reindex_mapping = {}
        for M in conj_tensor_network.tensors:
            for ind in M.inds:
                if ind not in contracted_inds:
                    reindex_mapping[ind] = conj_pfx + ind
        conj_tensor_network.reindex(reindex_mapping, inplace=True)
        partial_trace = conj_tensor_network @ tensor_network

        forward_inds = [self.i_str(self.qubit_map[keep_qubit]) for keep_qubit in keep_qubits]
        backward_inds = [conj_pfx + forward_ind for forward_ind in forward_inds]
        return partial_trace.to_dense(forward_inds, backward_inds)

    def to_numpy(self) -> np.ndarray:
        """An alias for the state vector."""
        return self.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        """Applies a unitary operation, mutating the object to represent the new state.

        op:
            The operation that mutates the object. Note that currently, only 1-
            and 2- qubit operations are currently supported.
        """

        old_inds = tuple([self.i_str(self.qubit_map[qubit]) for qubit in op.qubits])
        new_inds = tuple(['new_' + old_ind for old_ind in old_inds])

        U = protocols.unitary(op).reshape([qubit.dimension for qubit in op.qubits] * 2)
        U = qtn.Tensor(U, inds=(new_inds + old_inds))

        # TODO(tonybruguier): Explore using the Quimb's tensor network natively.

        if len(op.qubits) == 1:
            n = self.grouping[op.qubits[0]]

            self.M[n] = (U @ self.M[n]).reindex({new_inds[0]: old_inds[0]})
        elif len(op.qubits) == 2:
            n, p = [self.grouping[qubit] for qubit in op.qubits]

            if n == p:
                self.M[n] = (U @ self.M[n]).reindex(
                    {new_inds[0]: old_inds[0], new_inds[1]: old_inds[1]}
                )
            else:
                # This is the index on which we do the contraction. We need to add it iff it's
                # the first time that we do the joining for that specific pair.
                mu_ind = self.mu_str(n, p)
                if mu_ind not in self.M[n].inds:
                    self.M[n].new_ind(mu_ind)
                if mu_ind not in self.M[p].inds:
                    self.M[p].new_ind(mu_ind)

                T = U @ self.M[n] @ self.M[p]

                left_inds = tuple(set(T.inds) & set(self.M[n].inds)) + (new_inds[0],)
                X, Y = T.split(
                    left_inds,
                    method=self.simulation_options.method,
                    max_bond=self.simulation_options.max_bond,
                    cutoff=self.simulation_options.cutoff,
                    cutoff_mode=self.simulation_options.cutoff_mode,
                    get='tensors',
                    absorb='both',
                    bond_ind=mu_ind,
                )

                # Equations (13), (14), and (15):
                # TODO(tonybruguier): When Quimb 2.0.0 is released, the split()
                # function should have a 'renorm' that, when set to None, will
                # allow to compute e_n exactly as:
                # np.sum(abs((X @ Y).data) ** 2).real / np.sum(abs(T) ** 2).real
                #
                # The renormalization would then have to be done manually.
                #
                # However, for now, e_n are just the estimated value.
                e_n = self.simulation_options.cutoff
                self.estimated_gate_error_list.append(e_n)

                self.M[n] = X.reindex({new_inds[0]: old_inds[0]})
                self.M[p] = Y.reindex({new_inds[1]: old_inds[1]})
        else:
            # NOTE(tonybruguier): There could be a way to handle higher orders. I think this could
            # involve HOSVDs:
            # https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition
            #
            # TODO(tonybruguier): Evaluate whether it's even useful to implement and learn more
            # about HOSVDs.
            raise ValueError('Can only handle 1 and 2 qubit operations')

    def estimation_stats(self):
        "Returns some statistics about the memory usage and quality of the approximation."

        num_coefs_used = sum([Mi.data.size for Mi in self.M])
        memory_bytes = sum([Mi.data.nbytes for Mi in self.M])

        # The computation below is done for numerical stability, instead of directly using the
        # formula:
        # estimated_fidelity = \prod_i (1 - estimated_gate_error_list_i)
        estimated_fidelity = 1.0 + np.expm1(
            sum(np.log1p(-x) for x in self.estimated_gate_error_list)
        )
        estimated_fidelity = round(estimated_fidelity, ndigits=3)

        return {
            "num_coefs_used": num_coefs_used,
            "memory_bytes": memory_bytes,
            "estimated_fidelity": estimated_fidelity,
        }

    def perform_measurement(
        self, qubits: Sequence[ops.Qid], prng: np.random.RandomState, collapse_state_vector=True
    ) -> List[int]:
        """Performs a measurement over one or more qubits.

        Args:
            qubits: The sequence of qids to measure, in that order.
            prng: A random number generator, used to simulate measurements.
            collapse_state_vector: A Boolean specifying whether we should mutate
                the state after the measurement.
        """
        results: List[int] = []

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        for qubit in qubits:
            n = state.qubit_map[qubit]

            # Trace out other qubits
            M = state.partial_trace(keep_qubits={qubit})
            probs = np.diag(M).real
            sum_probs = sum(probs)

            # Because the computation is approximate, the probabilities do not
            # necessarily add up to 1.0, and thus we re-normalize them.
            if abs(sum_probs - 1.0) > self.simulation_options.sum_prob_atol:
                raise ValueError('Sum of probabilities exceeds tolerance: {}'.format(sum_probs))
            norm_probs = [x / sum_probs for x in probs]

            d = qubit.dimension
            result: int = int(prng.choice(d, p=norm_probs))

            collapser = np.zeros((d, d))
            collapser[result][result] = 1.0 / math.sqrt(probs[result])

            old_n = state.i_str(n)
            new_n = 'new_' + old_n

            collapser = qtn.Tensor(collapser, inds=(new_n, old_n))

            state.M[n] = (collapser @ state.M[n]).reindex({new_n: old_n})

            results.append(result)

        return results
