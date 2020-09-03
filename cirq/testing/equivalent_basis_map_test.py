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
import pytest

import cirq

from cirq import circuits


def test_correct_mappings():
    a, b, c = cirq.LineQubit.range(3)
    cirq.testing.assert_equivalent_computational_basis_map(
        maps={
            0b01: 0b01,
            0b10: 0b10
        },
        circuit=circuits.Circuit(cirq.IdentityGate(num_qubits=2).on(a, b)))

    cirq.testing.assert_equivalent_computational_basis_map(
        maps={
            0b001: 0b100,
            0b010: 0b010,
            0b100: 0b001,
        },
        circuit=circuits.Circuit(cirq.SWAP(a, c), cirq.I(b)))


def test_incorrect_mappings():
    a, b, c = cirq.LineQubit.range(3)
    with pytest.raises(AssertionError,
                       match=r"0b001 \(1\) was mapped to "
                       r"0b100 \(4\) instead of "
                       r"0b010 \(2\)"):
        cirq.testing.assert_equivalent_computational_basis_map(
            maps={
                0b001: 0b010,
                0b010: 0b100,
                0b100: 0b001,
            },
            circuit=circuits.Circuit(cirq.SWAP(a, c), cirq.I(b)))
