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
"""Remnants of Parallel two-qubit cross-entropy benchmarking on a grid.

This module keeps GridParallelXEBMetadata around for JSON backwards compatibility.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING
from cirq.experiments.random_quantum_circuit_generation import GridInteractionLayer
from cirq import protocols

if TYPE_CHECKING:
    import cirq


LAYER_A = GridInteractionLayer(col_offset=0, vertical=True, stagger=True)
LAYER_B = GridInteractionLayer(col_offset=1, vertical=True, stagger=True)
LAYER_C = GridInteractionLayer(col_offset=1, vertical=False, stagger=True)
LAYER_D = GridInteractionLayer(col_offset=0, vertical=False, stagger=True)


@dataclass(frozen=True, repr=False)
class GridParallelXEBMetadata:
    """Metadata for a grid parallel XEB experiment.

    Attributes:
        qubits: The qubits used in the experiment.
        two_qubit_gate: The entangling gate.
        num_circuits: Number of circuits.
        repetitions: Number of repetitions.
        cycles: Sequence of number of cycles.
        layers: Grid interaction layers.
        seed: pseudo-random number generator seed.
    """

    qubits: Sequence['cirq.Qid']
    two_qubit_gate: 'cirq.Gate'
    num_circuits: int
    repetitions: int
    cycles: Sequence[int]
    layers: Sequence[GridInteractionLayer]
    seed: Optional[int]

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)

    def __repr__(self) -> str:
        return (
            'cirq.experiments.randomized_and_cross_entropy_benchmarking.'
            'grid_parallel_two_qubit_xeb.GridParallelXEBMetadata('
            f'qubits={self.qubits!r}, '
            f'two_qubit_gate={self.two_qubit_gate!r}, '
            f'num_circuits={self.num_circuits!r}, '
            f'repetitions={self.repetitions!r}, '
            f'cycles={self.cycles!r}, '
            f'layers={self.layers!r}, '
            f'seed={self.seed!r})'
        )