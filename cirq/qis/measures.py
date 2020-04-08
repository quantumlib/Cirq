# Copyright 2019 The Cirq Developers
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
"""Measures on and between quantum states."""

import numpy as np
import scipy.stats


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calculates von Neumann entropy of density matrix in bits.
    Args:
        density_matrix: The density matrix.
    Returns:
        The calculated von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return scipy.stats.entropy(abs(eigenvalues), base=2)
