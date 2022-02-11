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
"""Objects and methods for acting efficiently on a state vector."""

from typing import Any, Optional, Tuple, TYPE_CHECKING, Type, Union, Dict, List, Sequence

import numpy as np

from cirq import _compat, linalg, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose
from cirq.linalg import transformations

if TYPE_CHECKING:
    import cirq
    from numpy.typing import DTypeLike


class _BufferedStateVector:
    def __init__(
        self,
        *,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0,
        qid_shape: Optional[Tuple[int, ...]] = None,
        available_buffer: Optional[np.ndarray] = None,
        dtype: Optional['DTypeLike'] = None,
    ):
        if qid_shape is not None:
            initial_tensor = qis.to_valid_state_vector(
                initial_state, len(qid_shape), qid_shape=qid_shape, dtype=dtype
            )
            if np.may_share_memory(initial_tensor, initial_state):
                initial_tensor = initial_tensor.copy()
            target_tensor = initial_tensor.reshape(qid_shape)
            self.target_tensor = target_tensor
        else:
            if isinstance(initial_state, np.ndarray):
                qid_shape = initial_state.shape
                self.target_tensor = initial_state
            else:
                qid_shape = ()
                self.target_tensor = qis.to_valid_state_vector(
                    initial_state, len(qid_shape), qid_shape=qid_shape, dtype=dtype
                )
        self.qid_shape = qid_shape
        self.available_buffer = (
            available_buffer if available_buffer is not None else np.empty_like(self.target_tensor)
        )

    def copy(self, deep_copy_buffers: bool = True) -> '_BufferedStateVector':
        target_tensor = self.target_tensor.copy()
        return _BufferedStateVector(
            initial_state=target_tensor,
            available_buffer=self.available_buffer.copy()
            if deep_copy_buffers
            else self.available_buffer,
        )

    def kron(self, other: '_BufferedStateVector') -> '_BufferedStateVector':
        target_tensor = transformations.state_vector_kronecker_product(
            self.target_tensor, other.target_tensor
        )
        return _BufferedStateVector(
            initial_state=target_tensor,
            available_buffer=np.empty_like(target_tensor),
        )

    def factor(
        self, axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['_BufferedStateVector', '_BufferedStateVector']:
        extracted_tensor, remainder_tensor = transformations.factor_state_vector(
            self.target_tensor, axes, validate=validate, atol=atol
        )
        extracted = _BufferedStateVector(
            initial_state=extracted_tensor,
            available_buffer=np.empty_like(extracted_tensor),
        )
        remainder = _BufferedStateVector(
            initial_state=remainder_tensor,
            available_buffer=np.empty_like(remainder_tensor),
        )
        return extracted, remainder

    def reindex(self, axes: Sequence[int]) -> '_BufferedStateVector':
        new_tensor = transformations.transpose_state_vector_to_axis_order(self.target_tensor, axes)
        return _BufferedStateVector(
            initial_state=new_tensor,
            available_buffer=np.empty_like(new_tensor),
        )

    def swap_target_tensor_for(self, new_target_tensor: np.ndarray):
        """Gives a new state vector for the system.

        Typically, the new state vector should be `args.available_buffer` where
        `args` is this `cirq.ActOnStateVectorArgs` instance.

        Args:
            new_target_tensor: The new system state. Must have the same shape
                and dtype as the old system state.
        """
        if new_target_tensor is self.available_buffer:
            self.available_buffer = self.target_tensor
        self.target_tensor = new_target_tensor

    def apply_unitary(self, unitary_value: Any, axes: Sequence[int]) -> bool:
        new_target_tensor = protocols.apply_unitary(
            unitary_value,
            protocols.ApplyUnitaryArgs(
                target_tensor=self.target_tensor,
                available_buffer=self.available_buffer,
                axes=axes,
            ),
            allow_decompose=False,
            default=NotImplemented,
        )
        if new_target_tensor is NotImplemented:
            return False
        self.swap_target_tensor_for(new_target_tensor)
        return True

    def apply_mixture(self, action: Any, axes: Sequence[int], prng) -> Tuple[bool, int]:
        mixture = protocols.mixture(action, default=None)
        if mixture is None:
            return False, 0
        probabilities, unitaries = zip(*mixture)

        index = prng.choice(range(len(unitaries)), p=probabilities)
        shape = protocols.qid_shape(action) * 2
        unitary = unitaries[index].astype(self.target_tensor.dtype).reshape(shape)
        linalg.targeted_left_multiply(unitary, self.target_tensor, axes, out=self.available_buffer)
        self.swap_target_tensor_for(self.available_buffer)
        return True, index

    def apply_channel(self, action: Any, axes: Sequence[int], prng) -> Tuple[bool, int]:
        kraus_operators = protocols.kraus(action, default=None)
        if kraus_operators is None:
            return False, 0

        def prepare_into_buffer(k: int):
            linalg.targeted_left_multiply(
                left_matrix=kraus_tensors[k],
                right_target=self.target_tensor,
                target_axes=axes,
                out=self.available_buffer,
            )

        shape = protocols.qid_shape(action)
        kraus_tensors = [
            e.reshape(shape * 2).astype(self.target_tensor.dtype) for e in kraus_operators
        ]
        p = prng.random()
        weight = None
        fallback_weight = 0
        fallback_weight_index = 0
        for index in range(len(kraus_tensors)):
            prepare_into_buffer(index)
            weight = np.linalg.norm(self.available_buffer) ** 2

            if weight > fallback_weight:
                fallback_weight_index = index
                fallback_weight = weight

            p -= weight
            if p < 0:
                break

        assert weight is not None, "No Kraus operators"
        if p >= 0 or weight == 0:
            # Floating point error resulted in a malformed sample.
            # Fall back to the most likely case.
            prepare_into_buffer(fallback_weight_index)
            weight = fallback_weight
            index = fallback_weight_index

        self.available_buffer /= np.sqrt(weight)
        self.swap_target_tensor_for(self.available_buffer)
        return True, index


class ActOnStateVectorArgs(ActOnArgs):
    """State and context for an operation acting on a state vector.

    There are two common ways to act on this object:

    1. Directly edit the `target_tensor` property, which is storing the state
        vector of the quantum system as a numpy array with one axis per qudit.
    2. Overwrite the `available_buffer` property with the new state vector, and
        then pass `available_buffer` into `swap_target_tensor_for`.
    """

    @_compat.deprecated_parameter(
        deadline='v0.15',
        fix='Use initial_state instead and specify all the arguments with keywords.',
        parameter_desc='target_tensor and positional arguments',
        match=lambda args, kwargs: 'target_tensor' in kwargs or len(args) != 1,
    )
    def __init__(
        self,
        target_tensor: Optional[np.ndarray] = None,
        available_buffer: Optional[np.ndarray] = None,
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, List[int]]] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0,
        dtype: Type[np.number] = np.complex64,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Inits ActOnStateVectorArgs.

        Args:
            target_tensor: The state vector to act on, stored as a numpy array
                with one dimension for each qubit in the system. Operations are
                expected to perform inplace edits of this object.
            available_buffer: A workspace with the same shape and dtype as
                `target_tensor`. Used by operations that cannot be applied to
                `target_tensor` inline, in order to avoid unnecessary
                allocations. Passing `available_buffer` into
                `swap_target_tensor_for` will swap it for `target_tensor`.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            initial_state: The initial state for the simulation in the
                computational basis.
            dtype: The `numpy.dtype` of the inferred state vector. One of
                `numpy.complex64` or `numpy.complex128`. Only used when
                `target_tenson` is None.
            classical_data: The shared classical data container for this
                simulation.
        """
        super().__init__(
            prng=prng,
            qubits=qubits,
            log_of_measurement_results=log_of_measurement_results,
            classical_data=classical_data,
        )
        self._state = _BufferedStateVector(
            initial_state=target_tensor if target_tensor is not None else initial_state,
            qid_shape=protocols.qid_shape(qubits) if qubits is not None else None,
            available_buffer=available_buffer,
            dtype=dtype,
        )

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        strats = [
            _strat_act_on_state_vector_from_apply_unitary,
            _strat_act_on_state_vector_from_mixture,
            _strat_act_on_state_vector_from_channel,
        ]
        if allow_decompose:
            strats.append(strat_act_on_from_apply_decompose)  # type: ignore

        # Try each strategy, stopping if one works.
        for strat in strats:
            result = strat(action, self, qubits)
            if result is False:
                break  # coverage: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)
        raise TypeError(
            "Can't simulate operations that don't implement "
            "SupportsUnitary, SupportsConsistentApplyUnitary, "
            "SupportsMixture or is a measurement: {!r}".format(action)
        )

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Delegates the call to measure the state vector."""
        bits, _ = sim.measure_state_vector(
            self.target_tensor,
            self.get_axes(qubits),
            out=self.target_tensor,
            qid_shape=self.target_tensor.shape,
            seed=self.prng,
        )
        return bits

    def _on_copy(self, target: 'cirq.ActOnStateVectorArgs', deep_copy_buffers: bool = True):
        target._state = self._state.copy(deep_copy_buffers)

    def _on_kronecker_product(
        self, other: 'cirq.ActOnStateVectorArgs', target: 'cirq.ActOnStateVectorArgs'
    ):
        target._state = self._state.kron(other._state)

    def _on_factor(
        self,
        qubits: Sequence['cirq.Qid'],
        extracted: 'cirq.ActOnStateVectorArgs',
        remainder: 'cirq.ActOnStateVectorArgs',
        validate=True,
        atol=1e-07,
    ):
        axes = self.get_axes(qubits)
        extracted._state, remainder._state = self._state.factor(axes, validate=validate, atol=atol)

    @property
    def allows_factoring(self):
        return True

    def _on_transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid'], target: 'cirq.ActOnStateVectorArgs'
    ):
        target._state = self._state.reindex(self.get_axes(qubits))

    def sample(
        self,
        qubits: Sequence['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self.qubit_map[q] for q in qubits]
        return sim.sample_state_vector(
            self.target_tensor,
            indices,
            qid_shape=tuple(q.dimension for q in self.qubits),
            repetitions=repetitions,
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            'cirq.ActOnStateVectorArgs('
            f'target_tensor={proper_repr(self.target_tensor)},'
            f' available_buffer={proper_repr(self.available_buffer)},'
            f' qubits={self.qubits!r},'
            f' log_of_measurement_results={proper_repr(self.log_of_measurement_results)})'
        )

    @property
    def target_tensor(self):
        return self._state.target_tensor

    @property
    def available_buffer(self):
        return self._state.available_buffer

    @property
    def qid_shape(self):
        return self._state.qid_shape


def _strat_act_on_state_vector_from_apply_unitary(
    action: Any, args: 'cirq.ActOnStateVectorArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    return True if args._state.apply_unitary(action, args.get_axes(qubits)) else NotImplemented


def _strat_act_on_state_vector_from_mixture(
    action: Any, args: 'cirq.ActOnStateVectorArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    ok, index = args._state.apply_mixture(action, args.get_axes(qubits), args.prng)
    if not ok:
        return NotImplemented
    if protocols.is_measurement(action):
        key = protocols.measurement_key_name(action)
        args._classical_data.record_channel_measurement(key, index)
    return True


def _strat_act_on_state_vector_from_channel(
    action: Any, args: 'cirq.ActOnStateVectorArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    ok, index = args._state.apply_channel(action, args.get_axes(qubits), args.prng)
    if not ok:
        return NotImplemented
    if protocols.is_measurement(action):
        key = protocols.measurement_key_name(action)
        args._classical_data.record_channel_measurement(key, index)
    return True
