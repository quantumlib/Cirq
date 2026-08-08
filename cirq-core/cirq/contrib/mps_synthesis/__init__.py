# Copyright 2025 The Cirq Developers
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

"""Synthesis for compiling a MPS to a circuit."""

from __future__ import annotations
from typing import TYPE_CHECKING

from cirq.contrib.mps_synthesis.mps_sequential import MPSSequential

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import cirq


def mps_circuit_from_statevector(
    statevector: npt.NDArray[np.complex128], max_num_layers: int = 10, target_fidelity: float = 0.95
) -> cirq.Circuit:
    """Create the circuit that encodes the statevector using MPS synthesis.

    Args:
        statevector: The target statevector to be encoded.
        max_num_layers: The maximum number of layers allowed in the circuit.
        target_fidelity: The target fidelity for the approximation.

    Returns:
        A cirq.Circuit that encodes the statevector.
    """
    encoder = MPSSequential(max_fidelity_threshold=target_fidelity)
    return encoder(statevector, max_num_layers=max_num_layers)
