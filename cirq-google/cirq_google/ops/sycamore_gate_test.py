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
import numpy as np
import pytest

import cirq
import cirq_google as cg


@pytest.mark.parametrize(
    'gate_type,qubit_count',
    (
        (cg.SYC, 2),
        (cirq.PhasedXPowGate(phase_exponent=0.1), 1),
        (cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.1), 1),
    ),
)
def test_consistent_protocols(gate_type, qubit_count):
    cirq.testing.assert_implements_consistent_protocols(
        gate_type,
        setup_code='import cirq\nimport numpy as np\nimport sympy\nimport cirq_google',
        qubit_count=qubit_count,
    )


def test_syc_str_repr():
    assert str(cg.SYC) == 'SYC'
    assert repr(cg.SYC) == 'cirq_google.SYC'


def test_syc_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cg.SYC(a, b))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───SYC───
      │
1: ───SYC───
""",
    )


def test_syc_is_specific_fsim():
    assert cg.SYC == cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)


def test_syc_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cg.SYC),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1j, 0],
                [0, -1j, 0, 0],
                [0, 0, 0, np.exp(-1j * np.pi / 6)],
            ]
        ),
        atol=1e-6,
    )
