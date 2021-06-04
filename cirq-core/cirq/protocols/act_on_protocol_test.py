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
import numpy as np
import cirq


class DummyActOnArgs(cirq.ActOnArgs):
    def __init__(self, implemented=False, measurements=None):
        super().__init__(np.random.RandomState())
        if measurements is None:
            measurements = []
        self.measurements = measurements
        self.implemented = implemented

    def _perform_measurement(self, qubits):
        return self.measurements

    def copy(self):
        return DummyActOnArgs(self.implemented, self.measurements.copy())

    def _act_on_fallback_(self, action, allow_decompose, qubits):
        return True if self.implemented else NotImplemented


def test_act_on_checks():
    args = DummyActOnArgs(True)
    cirq.act_on(cirq.X.on(cirq.LineQubit(0)), args)
    cirq.act_on_qubits(cirq.X, [cirq.LineQubit(0)], args)
