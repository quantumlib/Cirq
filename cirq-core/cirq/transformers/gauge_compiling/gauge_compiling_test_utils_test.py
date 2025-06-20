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


import numpy as np
import cirq
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester
from cirq.transformers import GaugeTransformer, GaugeSelector, ConstantGauge


class ExampleGate(cirq.testing.TwoQubitGate):
    unitary = cirq.unitary(cirq.CZ**0.123)

    def _unitary_(self) -> np.ndarray:
        return self.unitary


class ExampleSweepGate(cirq.testing.TwoQubitGate):
    unitary = cirq.unitary(cirq.CZ)

    def _unitary_(self) -> np.ndarray:
        return self.unitary


_EXAMPLE_TARGET = ExampleGate()
_EXAMPLE_SWEEP_TARGET = ExampleSweepGate()

_GOOD_TRANSFORMER = GaugeTransformer(
    target=_EXAMPLE_TARGET,
    gauge_selector=GaugeSelector(gauges=[ConstantGauge(two_qubit_gate=_EXAMPLE_TARGET)]),
)

_BAD_TRANSFORMER = GaugeTransformer(
    target=_EXAMPLE_TARGET,
    gauge_selector=GaugeSelector(
        gauges=[ConstantGauge(two_qubit_gate=_EXAMPLE_TARGET, pre_q0=cirq.X)]
    ),
)

_TRANSFORMER_WITH_SWEEP = GaugeTransformer(
    target=_EXAMPLE_SWEEP_TARGET,
    gauge_selector=GaugeSelector(
        gauges=[
            ConstantGauge(
                two_qubit_gate=_EXAMPLE_SWEEP_TARGET,
                pre_q0=cirq.Z,
                pre_q1=cirq.Z,
                post_q0=cirq.Z,
                post_q1=cirq.Z,
                support_sweep=True,
            )
        ]
    ),
)


class TestValidTransformer(GaugeTester):
    two_qubit_gate = _EXAMPLE_TARGET
    gauge_transformer = _GOOD_TRANSFORMER


class TestInvalidTransformer(GaugeTester):
    two_qubit_gate = _EXAMPLE_TARGET
    gauge_transformer = _BAD_TRANSFORMER
    must_fail = True


class TestSweep(GaugeTester):
    two_qubit_gate = _EXAMPLE_SWEEP_TARGET
    gauge_transformer = _TRANSFORMER_WITH_SWEEP
    sweep_must_pass = True
