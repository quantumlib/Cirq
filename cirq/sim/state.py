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


def pretty_state(state: Sequence, decimals: int=2) -> str:
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
        val = round(state[x], decimals)
        print(val)
        if round(val.real, decimals) == 0:
            val = val.imag
            format_str = "({:." + str(decimals) + "g}j)"
        if round(val.imag, decimals) == 0:
            val = val.real
        if val != 0:
            if round(state[x], decimals) == 1:
                components.append(ket.format(perm_list[x]))
            else:
                components.append((format_str + ket).format(val, perm_list[x]))

    return ' + '.join(components)


def decode_initial_state(initial_state: Union[int, np.ndarray],
    num_qubits: int, dtype: np.dtype = np.complex64) -> np.ndarray:
    """Verifies the initial_state is valid and converts it to ndarray form."""
    if isinstance(initial_state, np.ndarray):
        if len(initial_state) != 2 ** num_qubits:
            raise ValueError(
                'initial state was of size {} '
                'but expected state for {} qubits'.format(
                    len(initial_state), num_qubits))
        state = initial_state
    elif isinstance(initial_state, int):
        if initial_state < 0:
            raise ValueError('initial_state must be positive')
        elif initial_state >= 2 ** num_qubits:
            raise ValueError(
                'initial state was {} but expected state for {} qubits'.format(
                    initial_state, num_qubits))
        else:
            state = np.zeros(2 ** num_qubits, dtype=dtype)
            state[initial_state] = 1.0
    else:
        raise TypeError('initial_state was not of type int or ndarray')
    check_state(state, num_qubits)
    return state


def check_state(state: np.ndarray, num_qubits: int,
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
