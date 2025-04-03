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

import cirq
import cirq.contrib.qcircuit.qcircuit_pdf as qcircuit_pdf


def test_qcircuit_pdf_prepare_only():
    circuit = cirq.Circuit(cirq.X(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)))
    qcircuit_pdf.circuit_to_pdf_using_qcircuit_via_tex(
            circuit,
            "/tmp/test_file",
            prepare_only = True)
