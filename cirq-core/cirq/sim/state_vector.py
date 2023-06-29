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
"""Helpers for handling quantum state vectors."""

import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence

import numpy as np

from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils

if TYPE_CHECKING:
    import cirq


# For backwards compatibility and to make mypy happy:


class StateVectorMixin:
    """A mixin that provide methods for objects that have a state vector."""

    def __init__(self, qubit_map: Optional[Mapping['cirq.Qid', int]] = None, *args, **kwargs):
        """Inits StateVectorMixin.

        Args:
            qubit_map: A map from the Qubits in the Circuit to the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state (see the state_vector() method).
            *args: Passed on to the class that this is mixed in with.
            **kwargs: Passed on to the class that this is mixed in with.
        """
        super().__init__(*args, **kwargs)
        self._qubit_map = qubit_map or {}
        qid_shape = simulator._qubit_map_to_shape(self._qubit_map)
        self._qid_shape = None if qubit_map is None else qid_shape

    @property
    def qubit_map(self) -> Mapping['cirq.Qid', int]:
        return self._qubit_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        if self._qid_shape is None:
            return NotImplemented
        return self._qid_shape

    @abc.abstractmethod
    def state_vector(self, copy: bool = False) -> np.ndarray:
        """Return the state vector (wave function).

        The vector is returned in the computational basis with these basis
        states defined by the `qubit_map`. In particular the value in the
        `qubit_map` is the index of the qubit, and these are translated into
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
            copy: If True, the returned state vector will be a copy of that
            stored by the object. This is potentially expensive for large
            state vectors, but prevents mutation of the object state, e.g. for
            operating on intermediate states of a circuit.
            Defaults to False.
        """
        raise NotImplementedError()

    def dirac_notation(self, decimals: int = 2) -> str:
        """Returns the state vector as a string in Dirac notation.

        Args:
            decimals: How many decimals to include in the pretty print.

        Returns:
            A pretty string consisting of a sum of computational basis kets
            and non-zero floats of the specified accuracy."""
        return qis.dirac_notation(self.state_vector(), decimals, qid_shape=self._qid_shape)

    def density_matrix_of(self, qubits: Optional[List['cirq.Qid']] = None) -> np.ndarray:
        r"""Returns the density matrix of the state.

        Calculate the density matrix for the system on the qubits provided.
        Any qubits not in the list that are present in self.state_vector() will
        be traced out. If qubits is None, the full density matrix for
        self.state_vector() is returned, given self.state_vector() follows
        standard Kronecker convention of numpy.kron.

        For example, if `self.state_vector()` returns
        `np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)`,
        then `density_matrix_of(qubits = None)` gives us

        $$
        \rho = \begin{bmatrix}
                    0.5 & 0.5 \\
                    0.5 & 0.5
                \end{bmatrix}
        $$

        Args:
            qubits: list containing qubit IDs that you would like
                to include in the density matrix (i.e.) qubits that WON'T
                be traced out.

        Returns:
            A numpy array representing the density matrix.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if the indices are out of range for the number of qubits
                corresponding to the state.
        """
        return qis.density_matrix_from_state_vector(
            self.state_vector(),
            [self.qubit_map[q] for q in qubits] if qubits is not None else None,
            qid_shape=self._qid_shape,
        )

    def bloch_vector_of(self, qubit: 'cirq.Qid') -> np.ndarray:
        """Returns the bloch vector of a qubit in the state.

        Calculates the bloch vector of the given qubit
        in the state given by self.state_vector(), given that
        self.state_vector() follows the standard Kronecker convention of
        numpy.kron.

        Args:
            qubit: qubit who's bloch vector we want to find.

        Returns:
            A length 3 numpy array representing the qubit's bloch vector.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if index is out of range for the number of qubits
                corresponding to the state.
        """
        return qis.bloch_vector_from_state_vector(
            self.state_vector(), self.qubit_map[qubit], qid_shape=self._qid_shape
        )


def sample_state_vector(
    state_vector: np.ndarray,
    indices: Sequence[int],
    *,  # Force keyword args
    qid_shape: Optional[Tuple[int, ...]] = None,
    repetitions: int = 1,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> np.ndarray:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the passed in state.

    Args:
        state_vector: The multi-qubit state vector to be sampled. This is an
            array of 2 to the power of the number of qubit complex numbers, and
            so state must be of size ``2**integer``.  The `state_vector` can be
            a vector of size ``2**integer`` or a tensor of shape
            ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The `state_vector` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has its largest values in the 0th index.
        qid_shape: The qid shape of the `state_vector`.  Specify this argument
            when using qudits.
        repetitions: The number of times to sample.
        seed: A seed for the pseudorandom number generator.

    Returns:
        Measurement results with True corresponding to the ``|1⟩`` state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the supplied qubits. These lists
        are wrapped as a numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of `state_vector`
            is not a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the state.
    """
    if repetitions < 0:
        raise ValueError(f'Number of repetitions cannot be negative. Was {repetitions}')
    shape = qis.validate_qid_shape(state_vector, qid_shape)
    num_qubits = len(shape)
    qis.validate_indices(num_qubits, indices)

    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)), dtype=np.uint8)

    prng = value.parse_random_state(seed)

    # Calculate the measurement probabilities.
    probs = (state_vector * state_vector.conj()).real
    probs = simulation_utils.state_probabilities_by_indices(probs, indices, shape)

    # We now have the probability vector, correctly ordered, so sample over
    # it. Note that we us ints here, since numpy's choice does not allow for
    # choosing from a list of tuples or list of lists.
    result = prng.choice(len(probs), size=repetitions, p=probs)
    # Convert to individual qudit measurements.
    meas_shape = tuple(shape[i] for i in indices)
    return np.array(
        [value.big_endian_int_to_digits(result[i], base=meas_shape) for i in range(len(result))],
        dtype=np.uint8,
    )


def measure_state_vector(
    state_vector: np.ndarray,
    indices: Sequence[int],
    *,  # Force keyword args
    qid_shape: Optional[Tuple[int, ...]] = None,
    out: Optional[np.ndarray] = None,
    seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> Tuple[List[int], np.ndarray]:
    """Performs a measurement of the state in the computational basis.

    This does not modify `state` unless the optional `out` is `state`.

    Args:
        state_vector: The state to be measured. This state vector is assumed to
            be normalized. The state vector must be of size 2 ** integer.  The
            state vector can be of shape (2 ** integer) or (2, 2, ..., 2).
        indices: Which qubits are measured. The `state_vector` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has the largest values in the 0th index.
        qid_shape: The qid shape of the `state_vector`.  Specify this argument
            when using qudits.
        out: An optional place to store the result. If `out` is the same as
            the `state_vector` parameter, then `state_vector` will be modified
            inline. If `out` is not None, then the result is put into `out`.
            If `out` is None a new value will be allocated. In all of these
            case out will be the same as the returned ndarray of the method.
            The shape and dtype of `out` will match that of `state_vector` if
            `out` is None, otherwise it will match the shape and dtype of `out`.
        seed: A seed for the pseudorandom number generator.

    Returns:
        A tuple of a list and a numpy array. The list is an array of booleans
        corresponding to the measurement values (ordered by the indices). The
        numpy array is the post measurement state vector. This state vector has
        the same shape and dtype as the input `state_vector`.

    Raises:
        ValueError if the size of state is not a power of 2.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the state.
    """
    shape = qis.validate_qid_shape(state_vector, qid_shape)
    num_qubits = len(shape)
    qis.validate_indices(num_qubits, indices)

    if len(indices) == 0:
        if out is None:
            out = np.copy(state_vector)
        elif out is not state_vector:
            np.copyto(dst=out, src=state_vector)
        # Final else: if out is state then state will be modified in place.
        return ([], out)

    prng = value.parse_random_state(seed)

    # Cache initial shape.
    initial_shape = state_vector.shape

    # Calculate the measurement probabilities and then make the measurement.
    probs = (state_vector * state_vector.conj()).real
    probs = simulation_utils.state_probabilities_by_indices(probs, indices, shape)
    result = prng.choice(len(probs), p=probs)
    ###measurement_bits = [(1 & (result >> i)) for i in range(len(indices))]
    # Convert to individual qudit measurements.
    meas_shape = tuple(shape[i] for i in indices)
    measurement_bits = value.big_endian_int_to_digits(result, base=meas_shape)

    # Calculate the slice for the measurement result.
    result_slice = linalg.slice_for_qubits_equal_to(
        indices, big_endian_qureg_value=result, qid_shape=shape
    )

    # Create a mask which is False for only the slice.
    mask = np.ones(shape, dtype=bool)
    mask[result_slice] = False

    if out is None:
        out = np.copy(state_vector)
    elif out is not state_vector:
        np.copyto(dst=out, src=state_vector)
    # Final else: if out is state then state will be modified in place.

    # Potentially reshape to tensor, and then set masked values to 0.
    out.shape = shape
    out[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    out.shape = initial_shape
    out /= np.sqrt(probs[result])

    assert out is not None
    # We mutate and return out, so mypy cannot identify that the out cannot be None.
    return measurement_bits, out
