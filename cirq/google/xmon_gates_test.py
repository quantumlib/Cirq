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
import cirq.google as cg


def test_is_supported():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(1, 0)
    assert cg.is_native_xmon_op(cirq.CZ(a, b))
    assert cg.is_native_xmon_op(cirq.X(a)**0.5)
    assert cg.is_native_xmon_op(cirq.Y(a)**0.5)
    assert cg.is_native_xmon_op(cirq.Z(a)**0.5)
    assert cg.is_native_xmon_op(
        cirq.PhasedXPowGate(phase_exponent=0.2).on(a)**0.5)
    assert cg.is_native_xmon_op(cirq.Z(a)**1)
    assert not cg.is_native_xmon_op(cirq.CCZ(a, b, c))
    assert not cg.is_native_xmon_op(cirq.SWAP(a, b))
