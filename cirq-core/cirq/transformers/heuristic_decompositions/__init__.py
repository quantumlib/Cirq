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

"""Utilities for heuristic decomposition of cirq gates."""

from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
    TwoQubitGateTabulation as TwoQubitGateTabulation,
    TwoQubitGateTabulationResult as TwoQubitGateTabulationResult,
    two_qubit_gate_product_tabulation as two_qubit_gate_product_tabulation,
)
