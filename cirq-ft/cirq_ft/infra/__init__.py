# Copyright 2023 The Cirq Developers
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

from cirq_ft.infra.gate_with_registers import (
    GateWithRegisters,
    Register,
    Signature,
    Side,
    SelectionRegister,
    total_bits,
    split_qubits,
    merge_qubits,
    get_named_qubits,
)
from cirq_ft.infra.qubit_management_transformers import map_clean_and_borrowable_qubits
from cirq_ft.infra.t_complexity_protocol import TComplexity, t_complexity
