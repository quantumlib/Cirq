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
"""Helpers for handling quantum wavefunctions."""

import itertools

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import abc
import numpy as np

from cirq import linalg, ops


class StateVectorMixin():
    """A mixin that provide methods for objects that have a state vector.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering is
            used to define the state (see the state_vector() method).
    """

    # Reason for 'type: ignore': https://github.com/python/mypy/issues/5887
    def __init__(self, qubit_map: Optional[Dict[ops.Qid, int]] = None,
        *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self._qubit_map = qubit_map or {}

    @property
    def qubit_map(self):
        return self._qubit_map

    @abc.abstractmethod
    def state_vector(self) -> np.ndarray:
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

                    | QubitA | QubitB | QubitC
                :-: | :----: | :----: | :----:
                 0  |   0    |   0    |   0
                 1  |   0    |   0    |   1
                 2  |   0    |   1    |   0
                 3  |   0    |   1    |   1
                 4  |   1    |   0    |   0
                 5  |   1    |   0    |   1
                 6  |   1    |   1    |   0
                 7  |   1    |   1    |   1

        """
        raise NotImplementedError()

    def dirac_notation(self, decimals: int = 2) -> str:
        """Returns the state vector as a string in Dirac notation.

        Args:
            decimals: How many decimals to include in the pretty print.

        Returns:
            A pretty string consisting of a sum of computational basis kets
            and non-zero floats of the specified accuracy."""
        return dirac_notation(self.state_vector(), decimals)

    def density_matrix_of(self, qubits: List[ops.Qid] = None) -> np.ndarray:
        r"""Returns the density matrix of the state.

        Calculate the density matrix for the system on the list, qubits.
        Any qubits not in the list that are present in self.state_vector() will
        be traced out. If qubits is None the full density matrix for
        self.state_vector() is returned, given self.state_vector() follows
        standard Kronecker convention of numpy.kron.

        For example:
            self.state_vector() = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
                dtype=np.complex64)
            qubits = None
            gives us \rho = \begin{bmatrix}
                                0.5 & 0.5
                                0.5 & 0.5
                            \end{bmatrix}

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
        return density_matrix_from_state_vector(
            self.state_vector(),
            [self.qubit_map[q] for q in qubits] if qubits is not None else None
        )

    def bloch_vector_of(self, qubit: ops.Qid) -> np.ndarray:
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
        return bloch_vector_from_state_vector(self.state_vector(),
                                              self.qubit_map[qubit])


def bloch_vector_from_state_vector(state: Sequence, index: int) -> np.ndarray:
    """Returns the bloch vector of a qubit.

    Calculates the bloch vector of the qubit at index
    in the wavefunction given by state, assuming state follows
    the standard Kronecker convention of numpy.kron.

    Args:
        state: A sequence representing a wave function in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron.
        index: index of qubit who's bloch vector we want to find.
            follows the standard Kronecker convention of numpy.kron.

    Returns:
        A length 3 numpy array representing the qubit's bloch vector.

    Raises:
        ValueError: if the size of state is not a power of 2.
        ValueError: if the size of the state represents more than 25 qubits.
        IndexError: if index is out of range for the number of qubits
            corresponding to the state.
    """
    rho = density_matrix_from_state_vector(state, [index])
    v = np.zeros(3, dtype=np.float32)
    v[0] = 2*np.real(rho[0][1])
    v[1] = 2*np.imag(rho[1][0])
    v[2] = np.real(rho[0][0] - rho[1][1])

    return v


def density_matrix_from_state_vector(
    state: Sequence,
    indices: Iterable[int] = None
) -> np.ndarray:
    r"""Returns the density matrix of the wavefunction.

    Calculate the density matrix for the system on the given qubit
    indices, with the qubits not in indices that are present in state
    traced out. If indices is None the full density matrix for state
    is returned. We assume state follows the standard Kronecker
    convention of numpy.kron.

    For example:

        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
        indices = None

    gives us

        $$
        \rho = \begin{bmatrix}
                0.5 & 0.5
                0.5 & 0.5
            \end{bmatrix}
        $$

    Args:
        state: A sequence representing a wave function in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron.
        indices: list containing indices for qubits that you would like
            to include in the density matrix (i.e.) qubits that WON'T
            be traced out. follows the standard Kronecker convention of
            numpy.kron.

    Returns:
        A numpy array representing the density matrix.

    Raises:
        ValueError: if the size of state is not a power of 2.
        ValueError: if the size of the state represents more than 25 qubits.
        IndexError: if the indices are out of range for the number of qubits
            corresponding to the state.
    """
    n_qubits = _validate_num_qubits(state)

    if indices is None:
        return np.outer(state, np.conj(state))

    indices = list(indices)
    _validate_indices(n_qubits, indices)

    state = np.asarray(state).reshape((2,)*n_qubits)

    sum_inds = np.array(range(n_qubits))
    sum_inds[indices] += n_qubits

    rho = np.einsum(state, list(range(n_qubits)), np.conj(state),
        sum_inds.tolist(), indices + sum_inds[indices].tolist())
    new_shape = 2**len(indices)

    return rho.reshape((new_shape, new_shape))


def dirac_notation(state: Sequence, decimals: int=2) -> str:
    """Returns the wavefunction as a string in Dirac notation.

    For example:

        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
        print(dirac_notation(state)) -> 0.71|0⟩ + 0.71|1⟩

    Args:
        state: A sequence representing a wave function in which the ordering
            mapping to qubits follows the standard Kronecker convention of
            numpy.kron.
        decimals: How many decimals to include in the pretty print.

    Returns:
        A pretty string consisting of a sum of computational basis kets
        and non-zero floats of the specified accuracy.
    """
    perm_list = ["".join(seq) for seq in itertools.product(
        "01", repeat=int(len(state)).bit_length() - 1)]
    components = []
    ket = "|{}⟩"
    for x in range(len(perm_list)):
        format_str = "({:." + str(decimals) + "g})"
        val = (round(state[x].real, decimals)
               + 1j * round(state[x].imag, decimals))

        if round(val.real, decimals) == 0 and round(val.imag, decimals) != 0:
            val = val.imag
            format_str = "{:." + str(decimals) + "g}j"
        elif round(val.imag, decimals) == 0 and round(val.real, decimals) != 0:
            val = val.real
            format_str = "{:." + str(decimals) + "g}"
        if val != 0:
            if round(state[x].real, decimals) == 1 and \
               round(state[x].imag, decimals) == 0:
                components.append(ket.format(perm_list[x]))
            else:
                components.append((format_str + ket).format(val, perm_list[x]))
    if not components:
        return '0'

    return ' + '.join(components).replace(' + -', ' - ')


def to_valid_state_vector(state_rep: Union[int, np.ndarray],
                          num_qubits: int,
                          dtype: Type[np.number] = np.complex64) -> np.ndarray:
    """Verifies the state_rep is valid and converts it to ndarray form.

    This method is used to support passing in an integer representing a
    computational basis state or a full wave function as a representation of
    a state.

    Args:
        state_rep: If an int, the state returned is the state corresponding to
            a computational basis state. If an numpy array this is the full
            wave function. Both of these are validated for the given number
            of qubits, and the state must be properly normalized and of the
            appropriate dtype.
        num_qubits: The number of qubits for the state. The state_rep must be
            valid for this number of qubits.
        dtype: The numpy dtype of the state, will be used when creating the
            state for a computational basis state, or validated against if
            state_rep is a numpy array.

    Returns:
        A numpy ndarray corresponding to the state on the given number of
        qubits.

    Raises:
        ValueError if the state is not valid.
    """
    if isinstance(state_rep, np.ndarray):
        if len(state_rep) != 2 ** num_qubits:
            raise ValueError(
                'initial state was of size {} '
                'but expected state for {} qubits'.format(
                    len(state_rep), num_qubits))
        state = state_rep
    elif isinstance(state_rep, int):
        if state_rep < 0:
            raise ValueError('initial_state must be positive')
        elif state_rep >= 2 ** num_qubits:
            raise ValueError(
                'initial state was {} but expected state for {} qubits'.format(
                    state_rep, num_qubits))
        else:
            state = linalg.one_hot(shape=2**num_qubits,
                                   dtype=dtype,
                                   index=state_rep)
    else:
        raise TypeError('initial_state was not of type int or ndarray')
    validate_normalized_state(state, num_qubits, dtype)
    return state


def validate_normalized_state(state: np.ndarray,
                              num_qubits: int,
                              dtype: Type[np.number] = np.complex64) -> None:
    """Validates that the given state is a valid wave function."""
    if state.size != 1 << num_qubits:
        raise ValueError(
            'State has incorrect size. Expected {} but was {}.'.format(
                1 << num_qubits, state.size))
    if state.dtype != dtype:
        raise ValueError(
            'State has invalid dtype. Expected {} but was {}'.format(
                dtype, state.dtype))
    norm = np.sum(np.abs(state) ** 2)
    if not np.isclose(norm, 1):
        raise ValueError('State is not normalized instead had norm %s' % norm)


def sample_state_vector(state: np.ndarray,
                        indices: List[int],
                        repetitions: int=1) -> np.ndarray:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the passed in state.

    Args:
        state: The multi-qubit wavefunction to be sampled. This is an array of
            2 to the power of the number of qubit complex numbers, and so
            state must be of size ``2**integer``.  The state can be a vector of
            size ``2**integer`` or a tensor of shape ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The state is assumed to be supplied
            in big endian order. That is the xth index of v, when expressed as
            a bitstring, has its largest values in the 0th index.
        repetitions: The number of times to sample the state.

    Returns:
        Measurement results with True corresponding to the ``|1⟩`` state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the supplied qubits. These lists
        are wrapped as an numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of ``state`` is not
            a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the state.
    """
    if repetitions < 0:
        raise ValueError('Number of repetitions cannot be negative. Was {}'
                         .format(repetitions))
    num_qubits = _validate_num_qubits(state)
    _validate_indices(num_qubits, indices)

    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)))

    # Calculate the measurement probabilities.
    probs = _probs(state, indices, num_qubits)

    # We now have the probability vector, correctly ordered, so sample over
    # it. Note that we us ints here, since numpy's choice does not allow for
    # choosing from a list of tuples or list of lists.
    result = np.random.choice(len(probs), size=repetitions, p=probs)
    # Convert to bools and rearrange to match repetition being the outer list.
    return np.transpose([(1 & (result >> i)).astype(np.bool) for i in
                         range(len(indices))])


def measure_state_vector(
        state: np.ndarray,
        indices: List[int],
        out: np.ndarray = None) -> Tuple[List[bool], np.ndarray]:
    """Performs a measurement of the state in the computational basis.

    This does not modify `state` unless the optional `out` is `state`.

    Args:
        state: The state to be measured. This state is assumed to be normalized.
            The state must be of size 2 ** integer.  The state can be of shape
            (2 ** integer) or (2, 2, ..., 2).
        indices: Which qubits are measured. The state is assumed to be supplied
            in big endian order. That is the xth index of v, when expressed as
            a bitstring, has the largest values in the 0th index.
        out: An optional place to store the result. If `out` is the same as
            the `state` parameter, then state will be modified inline. If `out`
            is not None, then the result is put into `out`.  If `out` is None
            a new value will be allocated. In all of these case out will be the
            same as the returned ndarray of the method. The shape and dtype of
            `out` will match that of state if `out` is None, otherwise it will
            match the shape and dtype of `out`.

    Returns:
        A tuple of a list and an numpy array. The list is an array of booleans
        corresponding to the measurement values (ordered by the indices). The
        numpy array is the post measurement state. This state has the same
        shape and dtype as the input state.

    Raises:
        ValueError if the size of state is not a power of 2.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the state.
    """
    num_qubits = _validate_num_qubits(state)
    _validate_indices(num_qubits, indices)

    if len(indices) == 0:
        if out is None:
            out = np.copy(state)
        elif out is not state:
            np.copyto(dst=out, src=state)
        # Final else: if out is state then state will be modified in place.
        return ([], out)

    # Cache initial shape.
    initial_shape = state.shape

    # Calculate the measurement probabilities and then make the measurement.
    probs = _probs(state, indices, num_qubits)
    result = np.random.choice(len(probs), p=probs)
    measurement_bits = [(1 & (result >> i)) for i in range(len(indices))]

    # Calculate the slice for the measurement result.
    result_slice = linalg.slice_for_qubits_equal_to(indices, result)

    # Create a mask which is False for only the slice.
    mask = np.ones([2] * num_qubits, dtype=bool)
    mask[result_slice] = False

    if out is None:
        out = np.copy(state)
    elif out is not state:
        np.copyto(dst=out, src=state)
    # Final else: if out is state then state will be modified in place.

    # Potentially reshape to tensor, and then set masked values to 0.
    out.shape = [2] * num_qubits
    out[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    out.shape = initial_shape
    out /= np.sqrt(probs[result])

    return measurement_bits, out


def _probs(state: np.ndarray, indices: List[int],
           num_qubits: int) -> List[float]:
    """Returns the probabilities for a measurement on the given indices."""
    # Tensor of squared amplitudes, shaped a rank [2, 2, .., 2] tensor.
    tensor = np.reshape(state, [2] * num_qubits)

    # Calculate the probabilities for measuring the particular results.
    probs = [
        np.linalg.norm(
                tensor[linalg.slice_for_qubits_equal_to(indices, b)]) ** 2
        for b in range(2 ** len(indices))]

    # To deal with rounding issues, ensure that the probabilities sum to 1.
    probs /= sum(probs) # type: ignore
    return probs


def _validate_num_qubits(state: np.ndarray) -> int:
    """Validates that state's size is a power of 2, returning number of qubits.
    """
    size = state.size
    if size & (size - 1):
        raise ValueError('state.size ({}) is not a power of two.'.format(size))
    return size.bit_length() - 1


def _validate_indices(num_qubits: int, indices: List[int]) -> None:
    """Validates that the indices have values within range of num_qubits."""
    if any(index < 0 for index in indices):
        raise IndexError('Negative index in indices: {}'.format(indices))
    if any(index >= num_qubits for index in indices):
        raise IndexError('Out of range indices, must be less than number of '
                         'qubits but was {}'.format(indices))
