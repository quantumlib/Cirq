# Copyright 2022 The Cirq Developers
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

"""Gatesets which can act as compilation targets in Cirq."""

from cirq.transformers.target_gatesets.compilation_target_gateset import (
    create_transformer_with_kwargs as create_transformer_with_kwargs,
    CompilationTargetGateset as CompilationTargetGateset,
    TwoQubitCompilationTargetGateset as TwoQubitCompilationTargetGateset,
)

from cirq.transformers.target_gatesets.cz_gateset import CZTargetGateset as CZTargetGateset

from cirq.transformers.target_gatesets.sqrt_iswap_gateset import (
    SqrtIswapTargetGateset as SqrtIswapTargetGateset,
)
