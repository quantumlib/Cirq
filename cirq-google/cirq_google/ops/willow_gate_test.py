# Copyright 2025 The Cirq Developers
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
import cirq_google as cg


def test_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(
        cg.WILLOW,
        setup_code='import cirq\nimport numpy as np\nimport sympy\nimport cirq_google',
        qubit_count=2,
    )


def test_willow_str_repr():
    assert str(cg.WILLOW) == 'WILLOW'
    assert repr(cg.WILLOW) == 'cirq_google.WILLOW'


def test_willow_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cg.WILLOW(a, b))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───WILLOW───
      │
1: ───WILLOW───
""",
    )


def test_willow_is_specific_fsim():
    assert cg.WILLOW == cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 9)


def test_willow_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cg.WILLOW),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0j, 0.0],
                [0.0, -1.0j, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.exp(-1j * np.pi / 9)],
            ]
        ),
        atol=1e-6,
    )
