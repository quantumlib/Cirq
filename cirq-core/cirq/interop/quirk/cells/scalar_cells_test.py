# Copyright 2019 The Cirq Developers
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
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_scalar_operations() -> None:
    assert_url_to_circuit_returns('{"cols":[["…"]]}', cirq.Circuit())

    assert_url_to_circuit_returns(
        '{"cols":[["NeGate"]]}', cirq.Circuit(cirq.global_phase_operation(-1))
    )

    assert_url_to_circuit_returns('{"cols":[["i"]]}', cirq.Circuit(cirq.global_phase_operation(1j)))

    assert_url_to_circuit_returns(
        '{"cols":[["-i"]]}', cirq.Circuit(cirq.global_phase_operation(-1j))
    )

    assert_url_to_circuit_returns(
        '{"cols":[["√i"]]}', cirq.Circuit(cirq.global_phase_operation(1j**0.5))
    )

    assert_url_to_circuit_returns(
        '{"cols":[["√-i"]]}', cirq.Circuit(cirq.global_phase_operation(1j**-0.5))
    )
