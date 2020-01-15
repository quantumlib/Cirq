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
"""IQM devices https://iqm.fi/devices"""  # TODO: PQC-5

import cirq
from cirq import devices


class Adonis(devices.Device):
    """IQM's five-qubit superconducting device with pairwise connectivity.
    Details: https://iqm.fi/devices
    """
    # TODO: PQC-5

    def __init__(self):
        """Instantiate the description of an Adonis device"""
        qubit_diagram = "-Q-\n" \
                        "QQQ\n" \
                        "-Q-\n"
        self.qubits = cirq.GridQubit.from_diagram(qubit_diagram)
