# Copyright 2024 The Cirq Developers
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

import concurrent.futures
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from tqdm.notebook import tqdm


def _get_hamming_distance(bitstring_1: Sequence[int], bitstring_2: Sequence[int]) -> int:
    """Calculates the Hamming distance between two bitstrings.

    Args:
        bitstring_1: Bitstring 1
        bitstring_2: Bitstring 2

    Returns: The Hamming distance
    """
    return int(np.sum(~(np.array(bitstring_1) == np.array(bitstring_2))))


def _bitstrings_to_probs(bitstrings: npt.NDArray[np.int8]) -> Mapping[tuple, float]:
    """Given a list of bitstrings from different measurements returns a probability distribution.

    Args:
        bitstrings: The bitstring

    Returns:
    """
    probs_dict: dict[tuple, float] = {}

    num_shots = bitstrings.shape[0]
    for bits in bitstrings:
        bits_tuple = tuple(bits)
        if bits_tuple in probs_dict:
            probs_dict[bits_tuple] += 1 / num_shots
        else:
            probs_dict[bits_tuple] = 1 / num_shots

    return probs_dict


def _bitstring_format_helper(
    measured_bitstrings: npt.NDArray[np.int8], subsystem: Sequence[int] | None = None
) -> npt.NDArray[np.int8]:
    """Formats the bitstring for analysis based on the selected subsystem.

    Args:
        measured_bitstrings: Measured bitstring
        subsystem: Subsystem of interest

    Returns: The bitstring string for the subsystem
    """
    if subsystem is None:
        return measured_bitstrings

    return measured_bitstrings[:, :, subsystem]


def _compute_bitstring_purity(bitstrings: npt.NDArray[np.int8]) -> float:
    """Computes the purity of a bitstring.

    Args:
        bitstrings: The bitstrings measured using the same unitary operators

    Returns: The purity of the bitstring
    """

    probs = _bitstrings_to_probs(bitstrings)
    purity = 0
    for s, p in probs.items():
        for s_prime, p_prime in probs.items():
            purity += (-2.0) ** float(-_get_hamming_distance(s, s_prime)) * p * p_prime

    return purity * 2 ** (bitstrings.shape[-1])


def process_entropy_from_bitstrings(
    measured_bitstrings: npt.NDArray[np.int8],
    subsystem: tuple[int] | None = None,
    parallelize=False,
):
    bitstrings = _bitstring_format_helper(measured_bitstrings, subsystem)
    num_shots = bitstrings.shape[1]
    num_qubits = bitstrings.shape[-1]

    if num_shots == 1:
        return 0

    if parallelize:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            purities = list(executor.map(_compute_bitstring_purity, list(bitstrings)))
        purity = np.mean(purities)

    else:
        purity = np.mean([_compute_bitstring_purity(bitstring) for bitstring in tqdm(bitstrings)])

    purity_unbiased = purity * num_shots / (num_shots - 1) - (2**num_qubits) / (num_shots - 1)

    return purity_unbiased
