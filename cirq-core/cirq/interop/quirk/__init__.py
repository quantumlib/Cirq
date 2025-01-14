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
"""Code related to interoperating with Quirk, a drag-and-drop circuit simulator.

References:
    https://github.com/strilanc/quirk - Quirk source code.
    https://algassert.com/quirk - Live version of Quirk.
"""

# Imports from cells are only to ensure operation reprs work correctly.
from cirq.interop.quirk.cells import (
    QuirkArithmeticGate as QuirkArithmeticGate,
    QuirkInputRotationOperation as QuirkInputRotationOperation,
    QuirkQubitPermutationGate as QuirkQubitPermutationGate,
)

from cirq.interop.quirk.url_to_circuit import (
    quirk_json_to_circuit as quirk_json_to_circuit,
    quirk_url_to_circuit as quirk_url_to_circuit,
)
