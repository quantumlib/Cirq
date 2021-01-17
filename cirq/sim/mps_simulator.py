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
from typing import Any, cast, Dict, List, Iterator, Sequence

import numpy as np
import quimb.tensor as qtn

import cirq
from cirq import circuits, study, ops, protocols, value
from cirq.sim import simulator


class MPSSimulator(simulator.SimulatesSamples, simulator.SimulatesIntermediateState):
    """An efficient simulator for MPS circuits."""

    def __init__(self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None, rsum2_cutoff=1e-3):
        """Creates instance of `MPSSimulator`.

        Args:
            seed: The random seed to use for this simulator.
            rsum2_cutoff: We drop singular values so that the sum of the
                square of the dropped singular values divided by the sum of the
                square of all the singular values is less than rsum2_cutoff.
                This is related to the fidelity of the computation. If we have
                N 2D gates, then the estimated fidelity is
                (1 - rsum2_cutoff) ** N.
        """
        self.init = True
        self._prng = value.parse_random_state(seed)
        self.rsum2_cutoff = rsum2_cutoff

    def _base_iterator(
        self, circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList, initial_state: int
    ) -> Iterator['cirq.MPSSimulatorStepResult']:
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
                state=MPSState(qubit_map, self.rsum2_cutoff, initial_state=initial_state),
            )
            return

        state = MPSState(qubit_map, self.rsum2_cutoff, initial_state=initial_state)

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
    ) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        Args:
            inital_state: An integer specifying the inital
            state in the computational basis.
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
        final_simulator_state,
    ):

        return MPSTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, List[np.ndarray]]:

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


class MPSSimulatorStepResult(simulator.StepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

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


def _sum_reduce(M, ind):
    # TODO(tonybruguier): Use M.sum_reduce(ind, inplace=True) once
    # version 2.0 of Quimb is released.
    summer = qtn.Tensor([1.0] * M.ind_size(ind), inds=(ind,))
    Msum = M @ summer
    # Sometimes, we get a scalar, but we would rather keep it as tensor.
    if not isinstance(Msum, qtn.Tensor):
        Msum = qtn.Tensor(Msum)
    return Msum


@value.value_equality
class MPSState:
    """A state of the MPS simulation."""

    def __init__(self, qubit_map, rsum2_cutoff, initial_state=0):
        self.qubit_map = qubit_map
        self.M = []

        # The order of the qubits matters, because the state |01> is different from |10>. Since
        # Qimb uses strings to name tensor indices, we want to be able to sort them too. If we are
        # working with, say, 123 qubits then we want qubit 3 to come before qubit 100, but then
        # we want write the string '003' which comes before '100' in lexicographic order. The code
        # below is just simple string formatting.
        max_num_digits = len('%d' % (max(qubit_map.values())))
        self.format_i = 'i_%%.%dd' % (max_num_digits)
        self.format_mu = 'mu_%%.%dd_%%.%dd' % (max_num_digits, max_num_digits)

        for qubit in reversed(qubit_map.keys()):
            d = qubit.dimension
            x = np.zeros(d)
            x[initial_state % d] = 1.0

            i = qubit_map[qubit]
            self.M.append(qtn.Tensor(x, inds=(self.i_str(i),)))
            initial_state = initial_state // d
        self.M = self.M[::-1]
        self.rsum2_cutoff = rsum2_cutoff
        self.num_2d_gates = 0

    def i_str(self, i):
        return self.format_i % (i)

    def mu_str(self, i, j):
        # By convention, the lower index is always the first.
        i, j = min(i, j), max(i, j)
        return self.format_mu % (i, j)

    def __str__(self) -> str:
        return str([x.data for x in self.M])

    def _value_equality_values_(self) -> Any:
        return self.qubit_map, self.M, self.rsum2_cutoff

    def copy(self) -> 'MPSState':
        state = MPSState(self.qubit_map, self.rsum2_cutoff)
        state.M = [x.copy() for x in self.M]
        state.num_2d_gates = self.num_2d_gates
        return state

    def _sum_up(self, skip_tracing_out_for_qubits=None):
        M = qtn.Tensor(1.0)

        def _trace_out(i):
            return skip_tracing_out_for_qubits and (i not in skip_tracing_out_for_qubits)

        # We can aggregate the qubits in any order we want, theoretically. However, it's quite
        # possible to blow up the memory doing so. Instead, we do a greedy search through the
        # qubits that minimizes the memory required at every step.
        badness_queue = [(qubit, 0) for qubit in self.qubit_map.keys()]

        while len(badness_queue) > 0:
            # Update the badness:
            for idx in range(len(badness_queue)):
                qubit = badness_queue[idx][0]
                # We define the badness as the number of elements the tensor M post
                # aggregation and tracing out.
                i = self.qubit_map[qubit]
                new_inds = set(self.M[i].inds)
                uncollapsed_inds = new_inds - set(M.inds)

                badness = 1
                for ind, shape in zip(self.M[i].inds, self.M[i].shape):
                    if ind in uncollapsed_inds and not _trace_out(i):
                        badness *= shape

                badness_queue[idx] = (qubit, badness)
            badness_queue = sorted(badness_queue, key=lambda x: x[1])

            # Pop the element to aggregate
            qubit, _ = badness_queue.pop(0)

            i = self.qubit_map[qubit]
            M = M @ self.M[i]

            if _trace_out(i):
                M = _sum_reduce(M, self.i_str(i))

        return M

    def state_vector(self):
        M = self._sum_up()
        # Here, we rely on the formatting of the indices, and the fact that we have enough
        # leading zeros so that 003 comes before 100.
        sorted_ind = tuple(sorted(M.inds))
        return M.fuse({'i': sorted_ind}).data

    def to_numpy(self) -> np.ndarray:
        return self.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        U = protocols.unitary(op).reshape([qubit.dimension for qubit in op.qubits] * 2)

        if len(op.qubits) == 1:
            n = self.qubit_map[op.qubits[0]]

            old_n = self.i_str(n)
            new_n = 'new_' + old_n

            U = qtn.Tensor(U, inds=(new_n, old_n))
            self.M[n] = (U @ self.M[n]).reindex({new_n: old_n})
        elif len(op.qubits) == 2:
            self.num_2d_gates += 1

            n, p = [self.qubit_map[qubit] for qubit in op.qubits]

            old_n = self.i_str(n)
            old_p = self.i_str(p)
            new_n = 'new_' + old_n
            new_p = 'new_' + old_p

            U = qtn.Tensor(U, inds=(new_n, new_p, old_n, old_p))

            # This is the index on which we do the joining. We need to add it iff it's the first
            # time that we do the joining for that specific pair.
            mu_ind = self.mu_str(n, p)
            if mu_ind not in self.M[n].inds:
                self.M[n].new_ind(mu_ind)
            if mu_ind not in self.M[p].inds:
                self.M[p].new_ind(mu_ind)

            T = U @ self.M[n] @ self.M[p]

            left_inds = tuple(set(T.inds) & set(self.M[n].inds)) + (new_n,)
            X, Y = T.split(
                left_inds,
                cutoff=self.rsum2_cutoff,
                cutoff_mode='rsum2',
                get='tensors',
                absorb='both',
                bond_ind=mu_ind,
            )

            self.M[n] = X.reindex({new_n: old_n})
            self.M[p] = Y.reindex({new_p: old_p})
        else:
            # NOTE(tonybruguier): There could be a way to handle higher orders. I think this could
            # involve HOSVDs:
            # https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition
            #
            # TODO(tonybruguier): Evaluate whether it's even useful to implement and learn more
            # about HOSVDs.
            raise ValueError('Can only handle 1 and 2 qubit operations')

    def estimation_stats(self):
        num_coefs_used = sum([Mi.data.size for Mi in self.M])
        memory_bytes = sum([Mi.data.nbytes for Mi in self.M])

        # The computation below is done for numerical stability, instead of directly using the
        # formula:
        # estimated_fidelity = (1 - self.rsum2_cutoff) ** self.num_2d_gates
        estimated_fidelity = 1.0 + np.expm1(np.log1p(-self.rsum2_cutoff) * self.num_2d_gates)
        estimated_fidelity = round(estimated_fidelity, ndigits=3)

        return {
            "num_coefs_used": num_coefs_used,
            "memory_bytes": memory_bytes,
            "num_2d_gates": self.num_2d_gates,
            "estimated_fidelity": estimated_fidelity,
        }

    def perform_measurement(
        self, qubits: Sequence[ops.Qid], prng: np.random.RandomState, collapse_state_vector=True
    ) -> List[int]:
        results: List[int] = []

        qid_shape = [qubit.dimension for qubit in qubits]
        skip_tracing_out_for_qubits = {self.qubit_map[qubit] for qubit in qubits}

        M = self._sum_up(skip_tracing_out_for_qubits=skip_tracing_out_for_qubits)

        for i, qubit in enumerate(qubits):
            # Trace out other qubits
            M_traced_out = M
            for j in range(len(qubits)):
                if j != i:
                    M_traced_out = _sum_reduce(M_traced_out, self.i_str(self.qubit_map[qubits[j]]))

            probs = [abs(x) ** 2 for x in M_traced_out.data]

            # Because the computation is approximate, the probabilities do not
            # necessarily add up to 1.0, and thus we re-normalize them.
            norm_probs = [x / sum(probs) for x in probs]

            d = qubit.dimension
            result: int = int(prng.choice(d, p=norm_probs))

            collapser = np.zeros((d, d))
            collapser[result][result] = 1.0 / math.sqrt(probs[result])

            n = self.qubit_map[qubit]

            old_n = self.i_str(n)
            new_n = 'new_' + old_n

            collapser = qtn.Tensor(collapser, inds=(new_n, old_n))

            if collapse_state_vector:
                self.M[n] = (collapser @ self.M[n]).reindex({new_n: old_n})
            M = (collapser @ M).reindex({new_n: old_n})

            results.append(result)

        return results
