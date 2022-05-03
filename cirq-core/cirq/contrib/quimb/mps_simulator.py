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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import quimb.tensor as qtn

from cirq import devices, protocols, qis, value
from cirq._compat import deprecated_parameter
from cirq.sim import simulator, simulator_base
from cirq.sim.simulation_state import SimulationState

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
    simulator_base.SimulatorBase['MPSSimulatorStepResult', 'MPSTrialResult', 'MPSState']
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
        super().__init__(noise=noise, seed=seed)

    def _create_partial_simulation_state(
        self,
        initial_state: Union[int, 'MPSState'],
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> 'MPSState':
        """Creates MPSState args for simulating the Circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.

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
            classical_data=classical_data,
        )

    def _create_step_result(self, sim_state: 'cirq.SimulationStateBase[MPSState]'):
        return MPSSimulatorStepResult(sim_state)

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[MPSState]',
    ) -> 'MPSTrialResult':
        """Creates a single trial results with the measurements.

        Args:
            params: A ParamResolver for determining values of Symbols.
            measurements: A dictionary from measurement key (e.g. qubit) to the
                actual measurement array.
            final_simulator_state: The final state of the simulation.

        Returns:
            A single result.
        """
        return MPSTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )


class MPSTrialResult(simulator_base.SimulationTrialResultBase['MPSState']):
    """A single trial reult"""

    @simulator._deprecated_step_result_parameter(old_position=3)
    def __init__(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[MPSState]',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    @property
    def final_state(self) -> 'MPSState':
        return self._get_merged_sim_state()

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


class MPSSimulatorStepResult(simulator_base.StepResultBase['MPSState']):
    """A `StepResult` that can perform measurements."""

    def __init__(self, sim_state: 'cirq.SimulationStateBase[MPSState]'):
        """Results of a step of the simulator.
        Attributes:
            sim_state: The qubit:SimulationState lookup for this step.
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


@value.value_equality
class _MPSHandler(qis.QuantumStateRepresentation):
    """Quantum state of the MPS simulation."""

    def __init__(
        self,
        qid_shape: Tuple[int, ...],
        grouping: Dict[int, int],
        M: List[qtn.Tensor],
        format_i: str,
        estimated_gate_error_list: List[float],
        simulation_options: MPSOptions = MPSOptions(),
    ):
        """Creates an MPSQuantumState

        Args:
            qid_shape: Dimensions of the qubits represented.
            grouping: How to group qubits together, if None all are individual.
            M: The tensor list for maintaining the MPS state.
            format_i: A string for formatting the group labels.
            estimated_gate_error_list: The error estimations.
            simulation_options: Numerical options for the simulation.
        """
        self._qid_shape = qid_shape
        self._grouping = grouping
        self._M = M
        self._format_i = format_i
        self._format_mu = 'mu_{}_{}'
        self._simulation_options = simulation_options
        self._estimated_gate_error_list = estimated_gate_error_list

    @classmethod
    def create(
        cls,
        *,
        qid_shape: Tuple[int, ...],
        grouping: Dict[int, int],
        initial_state: int = 0,
        simulation_options: MPSOptions = MPSOptions(),
    ):
        """Creates an MPSQuantumState

        Args:
            qid_shape: Dimensions of the qubits represented.
            grouping: How to group qubits together, if None all are individual.
            initial_state: The initial computational basis state.
            simulation_options: Numerical options for the simulation.

        Raises:
            ValueError: If the grouping does not cover the qubits.
        """
        M = []
        for _ in range(max(grouping.values()) + 1):
            M.append(qtn.Tensor())

        # The order of the qubits matters, because the state |01> is different from |10>. Since
        # Quimb uses strings to name tensor indices, we want to be able to sort them too. If we are
        # working with, say, 123 qubits then we want qubit 3 to come before qubit 100, but then
        # we want write the string '003' which comes before '100' in lexicographic order. The code
        # below is just simple string formatting.
        max_num_digits = len(f'{max(grouping.values())}')
        format_i = f'i_{{:0{max_num_digits}}}'

        # TODO(tonybruguier): Instead of relying on sortable indices could you keep a parallel
        # mapping of e.g. qubit to string-index and do all "logic" on the qubits themselves and
        # only translate to string-indices when calling a quimb API.

        # TODO(tonybruguier): Refactor out so that the code below can also be used by
        # circuit_to_tensors in cirq.contrib.quimb.state_vector.

        for axis in reversed(range(len(qid_shape))):
            d = qid_shape[axis]
            x = np.zeros(d)
            x[initial_state % d] = 1.0

            n = grouping[axis]
            M[n] @= qtn.Tensor(x, inds=(format_i.format(axis),))
            initial_state = initial_state // d
        return _MPSHandler(
            qid_shape=qid_shape,
            grouping=grouping,
            M=M,
            format_i=format_i,
            estimated_gate_error_list=[],
            simulation_options=simulation_options,
        )

    def i_str(self, i: int) -> str:
        # Returns the index name for the i'th qid.
        return self._format_i.format(i)

    def mu_str(self, i: int, j: int) -> str:
        # Returns the index name for the pair of the i'th and j'th qids. Note
        # that by convention, the lower index is always the first in the output
        # string.
        smallest = min(i, j)
        largest = max(i, j)
        return self._format_mu.format(smallest, largest)

    def __str__(self) -> str:
        return str(qtn.TensorNetwork(self._M))

    def _value_equality_values_(self) -> Any:
        return self._qid_shape, self._M, self._simulation_options, self._grouping

    def copy(self, deep_copy_buffers: bool = True) -> '_MPSHandler':
        """Copies the object.

        Args:
            deep_copy_buffers: True by default, False to reuse the existing buffers.
        Returns:
            A copy of the object.
        """
        return _MPSHandler(
            simulation_options=self._simulation_options,
            grouping=self._grouping,
            qid_shape=self._qid_shape,
            M=[x.copy() for x in self._M],
            estimated_gate_error_list=self._estimated_gate_error_list.copy(),
            format_i=self._format_i,
        )

    def state_vector(self) -> np.ndarray:
        """Returns the full state vector.

        Returns:
            A vector that contains the full state.
        """
        tensor_network = qtn.TensorNetwork(self._M)
        state_vector = tensor_network.contract(inplace=False)

        # Here, we rely on the formatting of the indices, and the fact that we have enough
        # leading zeros so that 003 comes before 100.
        sorted_ind = tuple(sorted(state_vector.inds))
        return state_vector.fuse({'i': sorted_ind}).data

    def partial_trace(self, keep_axes: Set[int]) -> np.ndarray:
        """Traces out all qubits except keep_axes.

        Args:
            keep_axes: The set of axes that are left after computing the
                partial trace. For example, if we have a circuit for 3 qubits
                and this parameter only has one qubit, the entire density matrix
                would be 8x8, but this function returns a 2x2 matrix.

        Returns:
            An array that contains the partial trace.
        """

        contracted_inds = set(map(self.i_str, set(range(len(self._qid_shape))) - keep_axes))

        conj_pfx = "conj_"

        tensor_network = qtn.TensorNetwork(self._M)

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

        forward_inds = list(map(self.i_str, keep_axes))
        backward_inds = [conj_pfx + forward_ind for forward_ind in forward_inds]
        return partial_trace.to_dense(forward_inds, backward_inds)

    def to_numpy(self) -> np.ndarray:
        """An alias for the state vector."""
        return self.state_vector()

    def apply_op(self, op: Any, axes: Sequence[int], prng: np.random.RandomState):
        """Applies a unitary operation, mutating the object to represent the new state.

        op:
            The operation that mutates the object. Note that currently, only 1-
            and 2- qubit operations are currently supported.
        """

        old_inds = tuple(map(self.i_str, axes))
        new_inds = tuple(['new_' + old_ind for old_ind in old_inds])

        if protocols.has_unitary(op):
            U = protocols.unitary(op)
        else:
            mixtures = protocols.mixture(op)
            mixture_idx = int(prng.choice(len(mixtures), p=[mixture[0] for mixture in mixtures]))
            U = mixtures[mixture_idx][1]
        U = qtn.Tensor(
            U.reshape([self._qid_shape[axis] for axis in axes] * 2), inds=(new_inds + old_inds)
        )

        # TODO(tonybruguier): Explore using the Quimb's tensor network natively.

        if len(axes) == 1:
            n = self._grouping[axes[0]]

            self._M[n] = (U @ self._M[n]).reindex({new_inds[0]: old_inds[0]})
        elif len(axes) == 2:
            n, p = [self._grouping[axis] for axis in axes]

            if n == p:
                self._M[n] = (U @ self._M[n]).reindex(
                    {new_inds[0]: old_inds[0], new_inds[1]: old_inds[1]}
                )
            else:
                # This is the index on which we do the contraction. We need to add it iff it's
                # the first time that we do the joining for that specific pair.
                mu_ind = self.mu_str(n, p)
                if mu_ind not in self._M[n].inds:
                    self._M[n].new_ind(mu_ind)
                if mu_ind not in self._M[p].inds:
                    self._M[p].new_ind(mu_ind)

                T = U @ self._M[n] @ self._M[p]

                left_inds = tuple(set(T.inds) & set(self._M[n].inds)) + (new_inds[0],)
                X, Y = T.split(
                    left_inds,
                    method=self._simulation_options.method,
                    max_bond=self._simulation_options.max_bond,
                    cutoff=self._simulation_options.cutoff,
                    cutoff_mode=self._simulation_options.cutoff_mode,
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
                e_n = self._simulation_options.cutoff
                self._estimated_gate_error_list.append(e_n)

                self._M[n] = X.reindex({new_inds[0]: old_inds[0]})
                self._M[p] = Y.reindex({new_inds[1]: old_inds[1]})
        else:
            # NOTE(tonybruguier): There could be a way to handle higher orders. I think this could
            # involve HOSVDs:
            # https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition
            #
            # TODO(tonybruguier): Evaluate whether it's even useful to implement and learn more
            # about HOSVDs.
            raise ValueError('Can only handle 1 and 2 qubit operations')
        return True

    def estimation_stats(self):
        """Returns some statistics about the memory usage and quality of the approximation."""

        num_coefs_used = sum([Mi.data.size for Mi in self._M])
        memory_bytes = sum([Mi.data.nbytes for Mi in self._M])

        # The computation below is done for numerical stability, instead of directly using the
        # formula:
        # estimated_fidelity = \prod_i (1 - estimated_gate_error_list_i)
        estimated_fidelity = 1.0 + np.expm1(
            sum(np.log1p(-x) for x in self._estimated_gate_error_list)
        )
        estimated_fidelity = round(estimated_fidelity, ndigits=3)

        return {
            "num_coefs_used": num_coefs_used,
            "memory_bytes": memory_bytes,
            "estimated_fidelity": estimated_fidelity,
        }

    def _measure(
        self, axes: Sequence[int], prng: np.random.RandomState, collapse_state_vector=True
    ) -> List[int]:
        results: List[int] = []

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        for axis in axes:
            # Trace out other qubits
            M = state.partial_trace(keep_axes={axis})
            probs = np.diag(M).real
            sum_probs = sum(probs)

            # Because the computation is approximate, the probabilities do not
            # necessarily add up to 1.0, and thus we re-normalize them.
            if abs(sum_probs - 1.0) > self._simulation_options.sum_prob_atol:
                raise ValueError(f'Sum of probabilities exceeds tolerance: {sum_probs}')
            norm_probs = [x / sum_probs for x in probs]

            d = self._qid_shape[axis]
            result: int = int(prng.choice(d, p=norm_probs))

            collapser = np.zeros((d, d))
            collapser[result][result] = 1.0 / math.sqrt(probs[result])

            old_n = state.i_str(axis)
            new_n = 'new_' + old_n

            collapser = qtn.Tensor(collapser, inds=(new_n, old_n))

            state._M[axis] = (collapser @ state._M[axis]).reindex({new_n: old_n})

            results.append(result)

        return results

    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        """Measures the MPS.

        Args:
            axes: The axes to measure.
            seed: The random number seed to use.
        Returns:
            The measurements in axis order.
        """
        return self._measure(axes, value.parse_random_state(seed))

    def sample(
        self,
        axes: Sequence[int],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples the MPS.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """

        measurements: List[List[int]] = []
        prng = value.parse_random_state(seed)

        for _ in range(repetitions):
            measurements.append(self._measure(axes, prng, collapse_state_vector=False))

        return np.array(measurements, dtype=int)


@value.value_equality
class MPSState(SimulationState[_MPSHandler]):
    """A state of the MPS simulation."""

    @deprecated_parameter(
        deadline='v0.16',
        fix='Use kwargs instead of positional args',
        parameter_desc='args',
        match=lambda args, kwargs: len(args) > 1,
    )
    @deprecated_parameter(
        deadline='v0.16',
        fix='Replace log_of_measurement_results with'
        ' classical_data=cirq.ClassicalDataDictionaryStore(_records=logs).',
        parameter_desc='log_of_measurement_results',
        match=lambda args, kwargs: 'log_of_measurement_results' in kwargs,
    )
    def __init__(
        self,
        qubits: Sequence['cirq.Qid'],
        prng: np.random.RandomState,
        simulation_options: MPSOptions = MPSOptions(),
        grouping: Optional[Dict['cirq.Qid', int]] = None,
        initial_state: int = 0,
        log_of_measurement_results: Dict[str, Any] = None,
        classical_data: 'cirq.ClassicalDataStore' = None,
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
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: If the grouping does not cover the qubits.
        """
        qubit_map = {q: i for i, q in enumerate(qubits)}
        final_grouping = qubit_map if grouping is None else grouping
        if final_grouping.keys() != qubit_map.keys():
            raise ValueError('Grouping must cover exactly the qubits.')
        state = _MPSHandler.create(
            initial_state=initial_state,
            qid_shape=tuple(q.dimension for q in qubits),
            simulation_options=simulation_options,
            grouping={qubit_map[k]: v for k, v in final_grouping.items()},
        )
        if log_of_measurement_results is not None:
            super().__init__(
                state=state,
                prng=prng,
                qubits=qubits,
                log_of_measurement_results=log_of_measurement_results,
                classical_data=classical_data,
            )
        else:
            super().__init__(state=state, prng=prng, qubits=qubits, classical_data=classical_data)

    def i_str(self, i: int) -> str:
        # Returns the index name for the i'th qid.
        return self._state.i_str(i)

    def mu_str(self, i: int, j: int) -> str:
        # Returns the index name for the pair of the i'th and j'th qids. Note
        # that by convention, the lower index is always the first in the output
        # string.
        return self._state.mu_str(i, j)

    def __str__(self) -> str:
        return str(self._state)

    def _value_equality_values_(self) -> Any:
        return self.qubits, self._state

    def state_vector(self) -> np.ndarray:
        """Returns the full state vector.

        Returns:
            A vector that contains the full state.
        """
        return self._state.state_vector()

    def partial_trace(self, keep_qubits: Set['cirq.Qid']) -> np.ndarray:
        """Traces out all qubits except keep_qubits.

        Args:
            keep_qubits: The set of qubits that are left after computing the
                partial trace. For example, if we have a circuit for 3 qubits
                and this parameter only has one qubit, the entire density matrix
                would be 8x8, but this function returns a 2x2 matrix.

        Returns:
            An array that contains the partial trace.
        """
        return self._state.partial_trace(set(self.get_axes(list(keep_qubits))))

    def to_numpy(self) -> np.ndarray:
        """An alias for the state vector."""
        return self._state.to_numpy()

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        """Delegates the action to self.apply_op"""
        return self._state.apply_op(action, self.get_axes(qubits), self.prng)

    def estimation_stats(self):
        """Returns some statistics about the memory usage and quality of the approximation."""
        return self._state.estimation_stats()

    @property
    def M(self):
        return self._state._M
