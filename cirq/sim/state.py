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

from typing import Sequence, Union

import numpy as np


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
    dtype: np.dtype = np.complex64):
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
