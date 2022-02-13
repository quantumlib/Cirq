# Copyright 2021 The Cirq Developers
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
"""Objects and methods for acting efficiently on a density matrix."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Sequence, Type, Union

import numpy as np

from cirq import _compat, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose
from cirq.linalg import transformations

if TYPE_CHECKING:
    import cirq
    from numpy.typing import DTypeLike


class _BufferedDensityMatrix:
    def __init__(
        self,
        *,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0,
        qid_shape: Optional[Tuple[int, ...]] = None,
        available_buffer: Optional[List[np.ndarray]] = None,
        dtype: Optional['DTypeLike'] = None,
    ):
        if not isinstance(initial_state, np.ndarray):
            if qid_shape is None:
                raise ValueError(f'Must define qid_shape if initial_initial state is not ndarray')
            self.qid_shape = qid_shape
            self.target_tensor = qis.to_valid_density_matrix(
                initial_state, len(qid_shape), qid_shape=qid_shape, dtype=dtype
            ).reshape(qid_shape * 2)
        else:
            if qid_shape is not None:
                self.qid_shape = qid_shape
                self.target_tensor = initial_state.reshape(qid_shape * 2)
            else:
                target_shape = initial_state.shape
                if len(target_shape) % 2 != 0:
                    raise ValueError(
                        'The dimension of target_tensor is not divisible by 2.'
                        ' Require explicit qid_shape.'
                    )
                self.qid_shape = target_shape[: len(target_shape) // 2]
                self.target_tensor = initial_state

        if available_buffer is None:
            available_buffer = [np.empty_like(self.target_tensor) for _ in range(3)]
        self.available_buffer = available_buffer

    def copy(self, deep_copy_buffers: bool = True) -> '_BufferedDensityMatrix':
        target_tensor = self.target_tensor.copy()
        if deep_copy_buffers:
            available_buffer = [b.copy() for b in self.available_buffer]
        else:
            available_buffer = self.available_buffer
        return _BufferedDensityMatrix(
            initial_state=target_tensor,
            available_buffer=available_buffer,
        )

    def kron(self, other: '_BufferedDensityMatrix') -> '_BufferedDensityMatrix':
        target_tensor = transformations.density_matrix_kronecker_product(
            self.target_tensor, other.target_tensor
        )
        return _BufferedDensityMatrix(initial_state=target_tensor)

    def factor(
        self, axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['_BufferedDensityMatrix', '_BufferedDensityMatrix']:
        extracted_tensor, remainder_tensor = transformations.factor_density_matrix(
            self.target_tensor, axes, validate=validate, atol=atol
        )
        extracted = _BufferedDensityMatrix(initial_state=extracted_tensor)
        remainder = _BufferedDensityMatrix(initial_state=remainder_tensor)
        return extracted, remainder

    def reindex(self, axes: Sequence[int]) -> '_BufferedDensityMatrix':
        new_tensor = transformations.transpose_density_matrix_to_axis_order(
            self.target_tensor, axes
        )
        return _BufferedDensityMatrix(initial_state=new_tensor)

    def apply_channel(self, action: Any, axes: Sequence[int]) -> bool:
        """Apply channel to state."""
        result = protocols.apply_channel(
            action,
            args=protocols.ApplyChannelArgs(
                target_tensor=self.target_tensor,
                out_buffer=self.available_buffer[0],
                auxiliary_buffer0=self.available_buffer[1],
                auxiliary_buffer1=self.available_buffer[2],
                left_axes=axes,
                right_axes=[e + len(self.qid_shape) for e in axes],
            ),
            default=None,
        )
        if result is None:
            return False
        for i in range(len(self.available_buffer)):
            if result is self.available_buffer[i]:
                self.available_buffer[i] = self.target_tensor
        self.target_tensor = result
        return True


class ActOnDensityMatrixArgs(ActOnArgs):
    """State and context for an operation acting on a density matrix.

    To act on this object, directly edit the `target_tensor` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
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
        available_buffer: Optional[List[np.ndarray]] = None,
        qid_shape: Optional[Tuple[int, ...]] = None,
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, List[int]]] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        ignore_measurement_results: bool = False,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0,
        dtype: Type[np.number] = np.complex64,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Inits ActOnDensityMatrixArgs.

        Args:
            target_tensor: The state vector to act on, stored as a numpy array
                with one dimension for each qubit in the system. Operations are
                expected to perform inplace edits of this object.
            available_buffer: A workspace with the same shape and dtype as
                `target_tensor`. Used by operations that cannot be applied to
                `target_tensor` inline, in order to avoid unnecessary
                allocations.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            qid_shape: The shape of the target tensor.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            ignore_measurement_results: If True, then the simulation
                will treat measurement as dephasing instead of collapsing
                process. This is only applicable to simulators that can
                model dephasing.
            initial_state: The initial state for the simulation in the
                computational basis.
            dtype: The `numpy.dtype` of the inferred state vector. One of
                `numpy.complex64` or `numpy.complex128`. Only used when
                `target_tenson` is None.
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: The dimension of `target_tensor` is not divisible by 2
                and `qid_shape` is not provided.
        """
        if qubits is None:
            raise ValueError('qubits must be specified')
        super().__init__(
            prng=prng,
            qubits=qubits,
            log_of_measurement_results=log_of_measurement_results,
            ignore_measurement_results=ignore_measurement_results,
            classical_data=classical_data,
        )
        self._state = _BufferedDensityMatrix(
            initial_state=target_tensor if target_tensor is not None else initial_state,
            qid_shape=tuple(q.dimension for q in qubits),
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
            _strat_apply_channel_to_state,
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
            "SupportsMixture or SupportsKraus or is a measurement: {!r}".format(action)
        )

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Delegates the call to measure the density matrix."""
        bits, _ = sim.measure_density_matrix(
            self.target_tensor,
            self.get_axes(qubits),
            out=self.target_tensor,
            qid_shape=self.qid_shape,
            seed=self.prng,
        )
        return bits

    def _on_copy(self, target: 'cirq.ActOnDensityMatrixArgs', deep_copy_buffers: bool = True):
        target._state = self._state.copy(deep_copy_buffers)

    def _on_kronecker_product(
        self, other: 'cirq.ActOnDensityMatrixArgs', target: 'cirq.ActOnDensityMatrixArgs'
    ):
        target._state = self._state.kron(other._state)

    def _on_factor(
        self,
        qubits: Sequence['cirq.Qid'],
        extracted: 'cirq.ActOnDensityMatrixArgs',
        remainder: 'cirq.ActOnDensityMatrixArgs',
        validate=True,
        atol=1e-07,
    ):
        axes = self.get_axes(qubits)
        extracted._state, remainder._state = self._state.factor(axes, validate=validate, atol=atol)

    @property
    def allows_factoring(self):
        return True

    def _on_transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid'], target: 'cirq.ActOnDensityMatrixArgs'
    ):
        target._state = self._state.reindex(self.get_axes(qubits))

    def sample(
        self,
        qubits: Sequence['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self.qubit_map[q] for q in qubits]
        return sim.sample_density_matrix(
            self.target_tensor,
            indices,
            qid_shape=tuple(q.dimension for q in self.qubits),
            repetitions=repetitions,
            seed=seed,
        )

    @property
    def can_represent_mixed_states(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            'cirq.ActOnDensityMatrixArgs('
            f'target_tensor={proper_repr(self.target_tensor)},'
            f' available_buffer={proper_repr(self.available_buffer)},'
            f' qid_shape={self.qid_shape!r},'
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


def _strat_apply_channel_to_state(
    action: Any, args: 'cirq.ActOnDensityMatrixArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    """Apply channel to state."""
    return True if args._state.apply_channel(action, args.get_axes(qubits)) else NotImplemented
