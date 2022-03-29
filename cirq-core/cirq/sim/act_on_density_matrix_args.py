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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Type, Union

import numpy as np

from cirq import _compat, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose
from cirq.linalg import transformations

if TYPE_CHECKING:
    import cirq
    from numpy.typing import DTypeLike


class _BufferedDensityMatrix(qis.QuantumStateRepresentation):
    """Contains the density matrix and buffers for efficient state evolution."""

    def __init__(self, density_matrix: np.ndarray, buffer: Optional[List[np.ndarray]] = None):
        """Initializes the object with the inputs.

        This initializer creates the buffer if necessary.

        Args:
            density_matrix: The density matrix, must be correctly formatted. The data is not
                checked for validity here due to performance concerns.
            buffer: Optional, must be length 3 and same shape as the density matrix. If not
                provided, a buffer will be created automatically.
        Raises:
            ValueError: If the array is not the shape of a density matrix.
        """
        self._density_matrix = density_matrix
        if buffer is None:
            buffer = [np.empty_like(density_matrix) for _ in range(3)]
        self._buffer = buffer
        if len(density_matrix.shape) % 2 != 0:
            raise ValueError('The dimension of target_tensor is not divisible by 2.')
        self._qid_shape = density_matrix.shape[: len(density_matrix.shape) // 2]

    @classmethod
    def create(
        cls,
        *,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE'] = 0,
        qid_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional['DTypeLike'] = None,
        buffer: Optional[List[np.ndarray]] = None,
    ):
        """Creates a buffered density matrix with the requested state.

        Args:
            initial_state: The initial state for the simulation in the computational basis.
            qid_shape: The shape of the density matrix, if the initial state is provided as an int.
            dtype: The desired dtype of the density matrix.
            buffer: Optional, must be length 3 and same shape as the density matrix. If not
                provided, a buffer will be created automatically.
        Raises:
            ValueError: If initial state is provided as integer, but qid_shape is not provided.
        """
        if not isinstance(initial_state, np.ndarray):
            if qid_shape is None:
                raise ValueError('qid_shape must be provided if initial_state is not ndarray')
            density_matrix = qis.to_valid_density_matrix(
                initial_state, len(qid_shape), qid_shape=qid_shape, dtype=dtype
            ).reshape(qid_shape * 2)
        else:
            if qid_shape is not None:
                density_matrix = initial_state.reshape(qid_shape * 2)
            else:
                density_matrix = initial_state
            if np.may_share_memory(density_matrix, initial_state):
                density_matrix = density_matrix.copy()
        density_matrix = density_matrix.astype(dtype, copy=False)
        return cls(density_matrix, buffer)

    def copy(self, deep_copy_buffers: bool = True) -> '_BufferedDensityMatrix':
        """Copies the object.

        Args:
            deep_copy_buffers: True by default, False to reuse the existing buffers.
        Returns:
            A copy of the object.
        """
        return _BufferedDensityMatrix(
            density_matrix=self._density_matrix.copy(),
            buffer=[b.copy() for b in self._buffer] if deep_copy_buffers else self._buffer,
        )

    def kron(self, other: '_BufferedDensityMatrix') -> '_BufferedDensityMatrix':
        """Creates the Kronecker product with the other density matrix.

        Args:
            other: The density matrix with which to kron.
        Returns:
            The Kronecker product of the two density matrices.
        """
        density_matrix = transformations.density_matrix_kronecker_product(
            self._density_matrix, other._density_matrix
        )
        return _BufferedDensityMatrix(density_matrix=density_matrix)

    def factor(
        self, axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['_BufferedDensityMatrix', '_BufferedDensityMatrix']:
        """Factors out the desired axes.

        Args:
            axes: The axes to factor out. Only the left axes should be provided. For example, to
                extract [C,A] from density matrix of shape [A,B,C,D,A,B,C,D], `axes` should be
                [2,0], and the return value will be two density matrices ([C,A,C,A], [B,D,B,D]).
            validate: Perform a validation that the density matrix factors cleanly.
            atol: The absolute tolerance for the validation.
            Returns:
                A tuple with the `(extracted, remainder)` density matrices, where `extracted` means
                the sub-matrix which corresponds to the axes requested, and with the axes in the
                requested order, and where `remainder` means the sub-matrix on the remaining axes,
                in the same order as the original density matrix.
        """
        extracted_tensor, remainder_tensor = transformations.factor_density_matrix(
            self._density_matrix, axes, validate=validate, atol=atol
        )
        extracted = _BufferedDensityMatrix(density_matrix=extracted_tensor)
        remainder = _BufferedDensityMatrix(density_matrix=remainder_tensor)
        return extracted, remainder

    def reindex(self, axes: Sequence[int]) -> '_BufferedDensityMatrix':
        """Transposes the axes of a density matrix to a specified order.

        Args:
            axes: The desired axis order. Only the left axes should be provided. For example, to
                transpose [A,B,C,A,B,C] to [C,B,A,C,B,A], `axes` should be [2,1,0].
        Returns:
            The transposed density matrix.
        """
        new_tensor = transformations.transpose_density_matrix_to_axis_order(
            self._density_matrix, axes
        )
        return _BufferedDensityMatrix(density_matrix=new_tensor)

    def apply_channel(self, action: Any, axes: Sequence[int]) -> bool:
        """Apply channel to state.

        Args:
            action: The value with a channel to apply.
            axes: The axes on which to apply the channel.
        Returns:
            True if the action succeeded.
        """
        result = protocols.apply_channel(
            action,
            args=protocols.ApplyChannelArgs(
                target_tensor=self._density_matrix,
                out_buffer=self._buffer[0],
                auxiliary_buffer0=self._buffer[1],
                auxiliary_buffer1=self._buffer[2],
                left_axes=axes,
                right_axes=[e + len(self._qid_shape) for e in axes],
            ),
            default=None,
        )
        if result is None:
            return False
        for i in range(len(self._buffer)):
            if result is self._buffer[i]:
                self._buffer[i] = self._density_matrix
        self._density_matrix = result
        return True

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
        bits, _ = sim.measure_density_matrix(
            self._density_matrix,
            axes,
            out=self._density_matrix,
            qid_shape=self._qid_shape,
            seed=seed,
        )
        return bits

    def sample(
        self,
        axes: Sequence[int],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples the density matrix.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """
        return sim.sample_density_matrix(
            self._density_matrix,
            axes,
            qid_shape=self._qid_shape,
            repetitions=repetitions,
            seed=seed,
        )

    @property
    def supports_factor(self) -> bool:
        return True

    @property
    def can_represent_mixed_states(self) -> bool:
        return True


class ActOnDensityMatrixArgs(ActOnArgs):
    """State and context for an operation acting on a density matrix.

    To act on this object, directly edit the `target_tensor` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
    """

    @_compat.deprecated_parameter(
        deadline='v0.15',
        fix='Use classical_data.',
        parameter_desc='log_of_measurement_results and positional arguments',
        match=lambda args, kwargs: 'log_of_measurement_results' in kwargs or len(args) > 5,
    )
    @_compat.deprecated_parameter(
        deadline='v0.15',
        fix='Use cirq.dephase_measurements to transform the circuit before simulating.',
        parameter_desc='ignore_measurement_results',
        match=lambda args, kwargs: 'ignore_measurement_results' in kwargs or len(args) > 7,
    )
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
        state = _BufferedDensityMatrix.create(
            initial_state=target_tensor if target_tensor is not None else initial_state,
            qid_shape=tuple(q.dimension for q in qubits) if qubits is not None else None,
            dtype=dtype,
            buffer=available_buffer,
        )
        if ignore_measurement_results:
            super().__init__(
                state=state,
                prng=prng,
                qubits=qubits,
                log_of_measurement_results=log_of_measurement_results,
                ignore_measurement_results=ignore_measurement_results,
                classical_data=classical_data,
            )
        else:
            super().__init__(
                state=state,
                prng=prng,
                qubits=qubits,
                log_of_measurement_results=log_of_measurement_results,
                classical_data=classical_data,
            )
        self._state: _BufferedDensityMatrix = state

    def _act_on_fallback_(
        self,
        action: Any,
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        strats: List[Callable[[Any, Any, Sequence['cirq.Qid']], bool]] = [
            _strat_apply_channel_to_state,
        ]
        if allow_decompose:
            strats.append(strat_act_on_from_apply_decompose)

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
        return self._state._density_matrix

    @property
    def available_buffer(self):
        return self._state._buffer

    @property
    def qid_shape(self):
        return self._state._qid_shape


def _strat_apply_channel_to_state(
    action: Any, args: 'cirq.ActOnDensityMatrixArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    """Apply channel to state."""
    return True if args._state.apply_channel(action, args.get_axes(qubits)) else NotImplemented
