import numpy as np
import pytest

import cirq


@pytest.mark.parametrize('gate_type,qubit_count', (
    (cirq.SYC, 2),
    (cirq.PhasedXPowGate(phase_exponent=0.1), 1),
    (cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.1), 1),
))
def test_consistent_protocols(gate_type, qubit_count):
    cirq.testing.assert_implements_consistent_protocols(
        gate_type,
        setup_code=('import cirq\n'
                    'import numpy as np\n'
                    'import sympy\n'),
        qubit_count=qubit_count)


def test_syc_str_repr():
    assert str(cirq.SYC) == 'SYC'
    assert repr(cirq.SYC) == 'cirq.SycamoreGate()'


def test_syc_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.SYC(a, b))
    cirq.testing.assert_has_diagram(circuit, """
0: ───SYC───
      │
1: ───SYC───
""")


def test_syc_is_specific_fsim():
    assert cirq.SYC == cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)


def test_syc_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.SYC),
        np.array([
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, np.exp(-1j * np.pi / 6)],
        ]),
        atol=1e-6)
