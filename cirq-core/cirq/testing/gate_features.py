# Copyright 2021 The Cirq Developers
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

"""Simple gates used for testing purposes."""

from cirq.ops import raw_types


class SingleQubitGate(raw_types.Gate):
    """A gate that must be applied to exactly one qubit."""

    def _num_qubits_(self) -> int:
        return 1


class TwoQubitGate(raw_types.Gate):
    """A gate that must be applied to exactly two qubits."""

    def _num_qubits_(self) -> int:
        return 2


class ThreeQubitGate(raw_types.Gate):
    """A gate that must be applied to exactly three qubits."""

    def _num_qubits_(self) -> int:
        return 3


class DoesNotSupportSerializationGate(raw_types.Gate):
    """A gate that can't be serialized."""

    def __init__(self, n_qubits: int = 1):
        self.n_qubits = n_qubits

    def _num_qubits_(self) -> int:
        return self.n_qubits
