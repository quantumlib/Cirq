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

from typing import Sequence

import numpy as np

from cirq.linalg.transformations import targeted_left_multiply
from cirq import protocols


def compute_kraus_operations(initial_density_matrix: np.ndarray, noisy_circuit, qubits):
    """Computes all the density (Kraus) operators from a channel

    Note that this does not modify the density_matrix.

    Args:
        initial_density_matrix: The original density matrix.
        noisy_circuit: The (usually noisy) circuit.
        qubits: The list of qubits.

    Returns:
        A list of Kraus operators.
    """
    qubit_map = {q.with_dimension(1): i for i, q in enumerate(qubits)}

    kraus_operations = [initial_density_matrix]
    for op in noisy_circuit.all_operations():
        target_axes = [qubit_map[q.with_dimension(1)] for q in op.qubits]

        next_kraus_operations = []

        for op_kraus in protocols.channel(op, default=None):
            op_kraus_reshaped = np.conjugate(np.transpose(op_kraus)).reshape(
                [2] * (len(target_axes) * 2))
            for kraus_operation in kraus_operations:
                next_kraus_operation = targeted_left_multiply(
                    left_matrix=op_kraus_reshaped,
                    right_target=kraus_operation,
                    target_axes=target_axes)
                next_kraus_operations.append(next_kraus_operation)

        kraus_operations = next_kraus_operations
    return kraus_operations