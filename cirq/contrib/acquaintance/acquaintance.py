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

import cirq
from cirq.ops import gate_features

class AcquaintanceOpportunityGate(cirq.Gate, cirq.TextDiagrammable):
    """Represents an acquaintance opportunity."""

    def __repr__(self):
        return 'Acq'

    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        if args.known_qubit_count is None:
            return NotImplemented
        wire_symbol = '█' if args.use_unicode_characters else 'Acq'
        wire_symbols = (wire_symbol,) * args.known_qubit_count
        return gate_features.TextDiagramInfo(
                wire_symbols=wire_symbols)

ACQUAINT = AcquaintanceOpportunityGate()
