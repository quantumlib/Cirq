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

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from itertools import product
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


def _get_hamming_distance(
    bitstring_1: npt.NDArray[np.int8], bitstring_2: npt.NDArray[np.int8]
) -> int:
    """Calculates the Hamming distance between two bitstrings.
    Args:
        bitstring_1: Bitstring 1
        bitstring_2: Bitstring 2
    Returns: The Hamming distance
    """
    return (bitstring_1 ^ bitstring_2).sum().item()


def _bitstrings_to_probs(
    bitstrings: npt.NDArray[np.int8],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[Any]]:
    """Given a list of bitstrings from different measurements returns a probability distribution.
    Args:
        bitstrings: The bitstring
    Returns:
        A tuple of bitstrings and their corresponding probabilities.
    """

    num_shots = bitstrings.shape[0]
    unique_bitstrings, counts = np.unique(bitstrings, return_counts=True, axis=0)
    probs = counts / num_shots

    return (unique_bitstrings, probs)


def _bitstring_format_helper(
    measured_bitstrings: npt.NDArray[np.int8], subsystem: Sequence[int] | None = None
) -> npt.NDArray[np.int8]:
    """Formats the bitstring for analysis based on the selected subsystem.
    Args:
        measured_bitstrings: List of sampled measurement outcomes as a numpy array of bitstrings.
        subsystem: Subsystem of interest
    Returns: The bitstring string for the subsystem
    """
    if subsystem is None:
        return measured_bitstrings

    return measured_bitstrings[:, :, subsystem]


def _compute_bitstrings_contribution_to_purity(bitstrings: npt.NDArray[np.int8]) -> float:
    """Computes the contribution to the purity of the bitstrings.
    Args:
        bitstrings: The bitstrings measured using the same unitary operators
    Returns: The purity of the bitstring
    """

    bitstrings, probs = _bitstrings_to_probs(bitstrings)
    purity = 0
    for (s, p), (s_prime, p_prime) in product(zip(bitstrings, probs), repeat=2):
        purity += (-2.0) ** float(-_get_hamming_distance(s, s_prime)) * p * p_prime

    return purity * 2 ** (bitstrings.shape[-1])


def process_renyi_entropy_from_bitstrings(
    measured_bitstrings: npt.NDArray[np.int8],
    subsystem: tuple[int] | None = None,
    pool: Optional[ThreadPoolExecutor] = None,
) -> float:
    """Compute the Rényi entropy of an array of bitstrings.
    Args:
        measured_bitstrings: List of sampled measurement outcomes as a numpy array of bitstrings.
        subsystem: Subsystem of interest
        pool: ThreadPoolExecutor used to paralelleize the computation.

    Returns:
        A float indicating the computed entropy.
    """
    bitstrings = _bitstring_format_helper(measured_bitstrings, subsystem)
    num_shots = bitstrings.shape[1]
    num_qubits = bitstrings.shape[-1]

    if num_shots == 1:
        return 0

    if pool is not None:
        purities = list(pool.map(_compute_bitstrings_contribution_to_purity, list(bitstrings)))
        purity = np.mean(purities)

    else:
        purity = np.mean(
            [_compute_bitstrings_contribution_to_purity(bitstring) for bitstring in bitstrings]
        )

    purity_unbiased = purity * num_shots / (num_shots - 1) - (2**num_qubits) / (num_shots - 1)

    return -np.log2(purity_unbiased)
