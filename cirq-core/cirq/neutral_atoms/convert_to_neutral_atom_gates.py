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

from __future__ import annotations

from typing import TYPE_CHECKING

from cirq.neutral_atoms import neutral_atom_devices

if TYPE_CHECKING:
    import cirq


def is_native_neutral_atom_op(operation: cirq.Operation) -> bool:
    """Returns true if the operation is in the default neutral atom gateset."""
    return operation in neutral_atom_devices.neutral_atom_gateset()


def is_native_neutral_atom_gate(gate: cirq.Gate) -> bool:
    """Returns true if the gate is in the default neutral atom gateset."""
    return gate in neutral_atom_devices.neutral_atom_gateset()
