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
"""
Package for optimizers and gate compilers related to Google-specific devices.
"""
from cirq.google.optimizers.two_qubit_gates import (
    gate_product_tabulation,
    GateTabulation,
)

from cirq.google.optimizers.convert_to_sycamore_gates import (
    ConvertToSycamoreGates,)

from cirq.google.optimizers.convert_to_sqrt_iswap import (
    ConvertToSqrtIswapGates,)

from cirq.google.optimizers.convert_to_xmon_gates import (
    ConvertToXmonGates,)

from cirq.google.optimizers.optimize_for_sycamore import (
    optimized_for_sycamore,)

from cirq.google.optimizers.optimize_for_xmon import (
    optimized_for_xmon,)
