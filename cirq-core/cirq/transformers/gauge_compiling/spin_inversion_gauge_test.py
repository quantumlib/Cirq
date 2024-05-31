# Copyright 2024 The Cirq Developers
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
from cirq.transformers.gauge_compiling import SpinInversionGaugeTransformer
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester


class TestSpinInversionGauge_0(GaugeTester):
    two_qubit_gate = cirq.ZZ
    gauge_transformer = SpinInversionGaugeTransformer


class TestSpinInversionGauge_1(GaugeTester):
    two_qubit_gate = cirq.ZZ**0.1
    gauge_transformer = SpinInversionGaugeTransformer


class TestSpinInversionGauge_2(GaugeTester):
    two_qubit_gate = cirq.ZZ**-1
    gauge_transformer = SpinInversionGaugeTransformer


class TestSpinInversionGauge_3(GaugeTester):
    two_qubit_gate = cirq.ZZ**0.3
    gauge_transformer = SpinInversionGaugeTransformer
