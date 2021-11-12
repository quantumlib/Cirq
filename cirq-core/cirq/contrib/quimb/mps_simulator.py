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

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import numpy as np
import quimb.tensor as qtn

from cirq import devices, study, ops, protocols, value
from cirq.sim import simulator, simulator_base
from cirq.sim.act_on_args import ActOnArgs

if TYPE_CHECKING:
    import cirq


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
    simulator_base.SimulatorBase[
        'MPSSimulatorStepResult', 'MPSTrialResult', 'MPSState', 'MPSState'
    ],
):
    """An efficient simulator for MPS circuits."""

    def __init__(
        self,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        simulation_options: MPSOptions = MPSOptions(),
        grouping: Optional[Dict['cirq.Qid', int]] = None,
    ):
        """Creates instance of `MPSSimulator`.

        Args:
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.

        Raises:
            ValueError: If the noise model is not unitary or a mixture.
        """
        self.init = True
        noise_model = devices.NoiseModel.from_noise_model_like(noise)
        if not protocols.has_mixture(noise_model):
            raise ValueError(f'noise must be unitary or mixture but was {noise_model}')
        self.simulation_options = simulation_options
        self.grouping = grouping
        super().__init__(
            noise=noise,
            seed=seed,
        )

    def _create_partial_act_on_args(
        self,
        initial_state: Union[int, 'MPSState'],
        qubits: Sequence['cirq.Qid'],
        logs: Dict[str, Any],
    ) -> 'MPSState':
        """Creates MPSState args for simulating the Circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            logs: A mutable object that measurements are recorded into.

        Returns:
            MPSState args for simulating the Circuit.
        """
        if isinstance(initial_state, MPSState):
            return initial_state

        return MPSState(
            qubits=qubits,
            prng=self._prng,
            simulation_options=self.simulation_options,
            grouping=self.grouping,
            initial_state=initial_state,
            log_of_measurement_results=logs,
        )

    def _create_step_result(
        self,
        sim_state: 'cirq.OperationTarget[MPSState]',
    ):
        return MPSSimulatorStepResult(sim_state)

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: 'MPSSimulatorStepResult',
    ) -> 'MPSTrialResult':
        """Creates a single trial results with the measurements.

        Args:
            params: A ParamResolver for determining values of Symbols.
            measurements: A dictionary from measurement key (e.g. qubit) to the
                actual measurement array.
            final_step_result: The final step result of the simulation.

        Returns:
            A single result.
        """
        return MPSTrialResult(
            params=params, measurements=measurements, final_step_result=final_step_result
        )


class MPSTrialResult(simulator.SimulationTrialResult):
    """A single trial reult"""

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: 'MPSSimulatorStepResult',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_step_result=final_step_result
        )

    @property
    def final_state(self):
        return self._final_simulator_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._final_simulator_state
        return f'measurements: {samples}\noutput state: {final}'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('cirq.MPSTrialResult(...)')
        else:
            p.text(str(self))


class MPSSimulatorStepResult(simulator_base.StepResultBase['MPSState', 'MPSState']):
    """A `StepResult` that can perform measurements."""

    def __init__(
        self,
        sim_state: 'cirq.OperationTarget[MPSState]',
    ):
        """Results of a step of the simulator.
        Attributes:
            sim_state: The qubit:ActOnArgs lookup for this step.
        """
        super().__init__(sim_state)

    @property
    def state(self):
        return self._merged_sim_state

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

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text("cirq.MPSSimulatorStepResult(...)" if cycle else self.__str__())

    def _simulator_state(self):
        return self.state


@value.value_equality
class MPSState(ActOnArgs):
    """A state of the MPS simulation."""

    def __init__(
        self,
        qubits: Sequence['cirq.Qid'],
        prng: np.random.RandomState,
        simulation_options: MPSOptions = MPSOptions(),
        grouping: Optional[Dict['cirq.Qid', int]] = None,
        initial_state: int = 0,
        log_of_measurement_results: Dict[str, Any] = None,
    ):
        """Creates and MPSState

        Args:
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: A random number generator, used to simulate measurements.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.
            initial_state: An integer representing the initial state.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.

        Raises:
            ValueError: If the grouping does not cover the qubits.
        """
        super().__init__(prng, qubits, log_of_measurement_results)
        qubit_map = self.qubit_map
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
        max_num_digits = len(f'{max(qubit_map.values())}')
        self.format_i = f'i_{{:0{max_num_digits}}}'
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

    def _on_copy(self, target: 'MPSState'):
        target.simulation_options = self.simulation_options
        target.grouping = self.grouping
        target.M = [x.copy() for x in self.M]
        target.estimated_gate_error_list = self.estimated_gate_error_list

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

    def apply_op(self, op: 'cirq.Operation', prng: np.random.RandomState):
        """Applies a unitary operation, mutating the object to represent the new state.

        op:
            The operation that mutates the object. Note that currently, only 1-
            and 2- qubit operations are currently supported.
        """

        old_inds = tuple([self.i_str(self.qubit_map[qubit]) for qubit in op.qubits])
        new_inds = tuple(['new_' + old_ind for old_ind in old_inds])

        if protocols.has_unitary(op):
            U = protocols.unitary(op)
        else:
            mixtures = protocols.mixture(op)
            mixture_idx = int(prng.choice(len(mixtures), p=[mixture[0] for mixture in mixtures]))
            U = mixtures[mixture_idx][1]
        U = qtn.Tensor(
            U.reshape([qubit.dimension for qubit in op.qubits] * 2), inds=(new_inds + old_inds)
        )

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
        return True

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        """Delegates the action to self.apply_op"""
        if isinstance(action, ops.Gate):
            action = ops.GateOperation(action, qubits)
        return self.apply_op(action, self.prng)

    def estimation_stats(self):
        """Returns some statistics about the memory usage and quality of the approximation."""

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

        Raises:
            ValueError: If the probabilities for the measurements differ too much from one for the
                tolerance specified in simulation options.
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
                raise ValueError(f'Sum of probabilities exceeds tolerance: {sum_probs}')
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

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Measures the axes specified by the simulator."""
        return self.perform_measurement(qubits, self.prng)

    def sample(
        self,
        qubits: Sequence[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:

        measurements: List[List[int]] = []

        for _ in range(repetitions):
            measurements.append(
                self.perform_measurement(
                    qubits, value.parse_random_state(seed), collapse_state_vector=False
                )
            )

        return np.array(measurements, dtype=int)
