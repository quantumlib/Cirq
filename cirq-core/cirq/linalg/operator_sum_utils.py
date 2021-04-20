# Copyright 2020 The Cirq Developers
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
"""Utils for the computation of operator sum (Kraus operators)."""

from typing import Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq.linalg.transformations import targeted_left_multiply

if TYPE_CHECKING:
    import cirq


def compute_kraus_operations(
    initial_density_matrix: np.ndarray,
    operations: Sequence[Tuple[Sequence[np.ndarray], Sequence[int]]],
):
    """Computes all the density (Kraus) operators from a channel

    Note that this does not modify the density_matrix.

    Args:
        initial_density_matrix: The original density matrix.
        operations: Tuple with first element being the operation matrice and the second the axes.

    Returns:
        A list of Kraus operators.
    """
    kraus_operations = [initial_density_matrix]
    for operation in operations:
        target_axes = operation[1]

        next_kraus_operations = []

        for op_kraus_reshaped in operation[0]:
            for kraus_operation in kraus_operations:
                next_kraus_operation = targeted_left_multiply(
                    left_matrix=op_kraus_reshaped,
                    right_target=kraus_operation,
                    target_axes=target_axes,
                )
                next_kraus_operations.append(next_kraus_operation)

        kraus_operations = next_kraus_operations
    return kraus_operations
