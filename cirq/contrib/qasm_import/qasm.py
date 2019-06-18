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

from cirq import circuits
from cirq.contrib.qasm_import._parser import QasmParser


class QasmCircuitParser:
    """QasmCircuitParser is currently partially developed, not functional,
    DO NOT USE.
    TODO(https://github.com/quantumlib/Cirq/issues/1548)
    It will serve as the entrypoint for parsing QASM files."""

    def __init__(self):
        pass

    def parse(self, qasm: str) -> circuits.Circuit:
        return QasmParser().parse(qasm).circuit
