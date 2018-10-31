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
"""Helpers for handling states."""

import itertools

from typing import List, Sequence, Tuple, Union

import numpy as np

from cirq import linalg


def dirac_notation(state: Sequence, decimals: int=2) -> str:
    """Returns the wavefunction as a string in Dirac notation.

    For example:
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
        print(pretty_state(state)) -> 0.71|0⟩ + 0.71|1⟩

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
        # Python 2 rounds imaginary numbers to 0, so need to round separately.
        val = (round(state[x].real, decimals)
               + 1j * round(state[x].imag, decimals))
        if round(val.real, decimals) == 0 and round(val.imag, decimals) != 0:
            val = val.imag
            format_str = "({:." + str(decimals) + "g}j)"
        if round(val.imag, decimals) == 0 and round(val.real, decimals) != 0:
            val = val.real
        if val != 0:
            if round(state[x], decimals) == 1:
                components.append(ket.format(perm_list[x]))
            else:
                components.append((format_str + ket).format(val, perm_list[x]))

    return ' + '.join(components)


def to_valid_state_vector(state_rep: Union[int, np.ndarray],
    num_qubits: int, dtype: np.dtype = np.complex64) -> np.ndarray:
    """Verifies the initial_state is valid and converts it to ndarray form.

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
            state = np.zeros(2 ** num_qubits, dtype=dtype)
            state[state_rep] = 1.0
    else:
        raise TypeError('initial_state was not of type int or ndarray')
    validate_normalized_state(state, num_qubits)
    return state


def validate_normalized_state(state: np.ndarray, num_qubits: int,
    dtype: np.dtype = np.complex64) -> None:
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


def sample_state_vector(
    state: np.ndarray,
    indices: List[int],
    repetitions: int=1) -> List[List[bool]]:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the passed in state.

    Args:
        state: The state to be measured. This state is assumed to be normalized.
            The state must be of size 2 ** integer.  The state can be of shape
            (2 ** integer) or (2, 2, ..., 2).
        indices: Which qubits are measured. The state is assumed to be supplied
            in big endian order. That is the xth index of v, when expressed as
            a bitstring, has the largest values that the 0th index.
        repetitions: The number of times to sample the state.

    Returns:
        Measurement results with True corresponding to the |1> state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the input indices.

    Raises:
        ValueError if repetitions is less than one or size of state is not a
            power of 2.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the state.
    """
    if repetitions < 1:
        raise ValueError(
            'Number of repetitions cannot be negative. Was {}'.format(
                repetitions))
    num_qubits = _validate_num_qubits(state)
    _validate_indices(num_qubits, indices)

    if len(indices) == 0:
        return [[]]

    # Calculate the measurement probabilities.
    probs = _probs(state, indices, num_qubits)

    # We now have the probability vector, correctly ordered, so sample over
    # it. Note that we us ints here, since numpy's choice does not allow for
    # choosing from a list of tuples or list of lists.
    result = np.random.choice(len(probs), size=repetitions, p=probs)
    # Convert to bools and rearrange to match repetition being the outer list.
    return np.transpose([(1 & (result >> i)).astype(np.bool) for i in
                         range(len(indices))]).tolist()


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
        out: An optional place to store the result. If out is the same as
            the `state` parameter, then state will be modified inline. If out
            is not None, then the result is put into out.  If out is None
            a new value will be allocated. In all of these case out will be the
            same as the returned ndarray of the method. The shape and dtype of
            out will always match that of state.

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
        return ([], np.copy(state))

    # Cache initial state.
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
        np.copyto(out, state)
    # Final else: if out is state then state will be modified in place.

    # Potentially reshape to tensor, and then set masked values to 0.
    out.shape = [2] * num_qubits
    out[mask] = 0

    # Restore original shape (if necessary) and renormalize.
    out.shape = initial_shape
    out /= np.sqrt(probs[result])

    return measurement_bits, out


def _probs(state: np.ndarray, indices: List[int],
           num_qubits: int):
    """Returns the probabilities for a measurement on the given indices."""
    # Tensor of squared amplitudes, shaped a rank [2, 2, .., 2] tensor.
    prob_tensor = np.abs(np.reshape(state, [2] * num_qubits)) ** 2

    # Calculate the probabilities for measuring the particular results.
    probs = [np.sum(prob_tensor[linalg.slice_for_qubits_equal_to(indices, b)])
             for b in range(2 ** len(indices))]
    # To deal with rounding issues, ensure that the probabilities sum to 1.
    probs /= sum(probs) # type: ignore
    return probs


def _validate_num_qubits(state: np.ndarray) -> int:
    """Validates that state's size is a power of 2, returning number of qubits.
    """
    size = state.size
    if size != 0 and size & (size - 1):
        raise ValueError(
            'state.size ({}) is not a power of two.'.format(size))
    return size.bit_length() - 1


def _validate_indices(num_qubits: int, indices: List[int]) -> None:
    """Validates that the indices have values within range of num_qubits."""
    if any(index < 0 for index in indices):
        raise IndexError('Negative index in indices: {}'.format(indices))
    if any(index >= num_qubits for index in indices):
        raise IndexError('Out of range indices, must be less than number of '
                         'qubits but was {}'.format(indices))