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
import pytest
import sympy

import cirq

# Import the class directly for testing
from cirq.contrib.quantikz.circuit_to_latex_quantikz import (
    CircuitToQuantikz,
    DEFAULT_PREAMBLE_TEMPLATE,
)


def test_empty_circuit_raises_value_error():
    """Test that an empty circuit raises a ValueError."""
    empty_circuit = cirq.Circuit()
    with pytest.raises(ValueError, match="Input circuit cannot be empty."):
        CircuitToQuantikz(empty_circuit)


def test_circuit_no_qubits_raises_value_error():
    """Test that a circuit with no qubits raises a ValueError."""
    empty_circuit = cirq.Circuit(cirq.global_phase_operation(-1))
    with pytest.raises(ValueError, match="Circuit contains no qubits."):
        CircuitToQuantikz(empty_circuit)


def test_basic_circuit_conversion():
    """Test a simple circuit conversion to LaTeX."""
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.PhasedXZGate(x_exponent=1, z_exponent=1, axis_phase_exponent=0.5)(q0),
        cirq.CZ(q0, q1),
        cirq.CNOT(q0, q1),
        cirq.SWAP(q0, q1),
        cirq.Moment(cirq.CZ(q1, q2), cirq.CZ(q0, q3)),
        cirq.FSimGate(np.pi / 2, np.pi / 6)(q0, q1),
        cirq.measure(q0, q2, q3, key='m0'),
    )
    converter = CircuitToQuantikz(circuit, wire_labels="q")
    latex_code = converter.generate_latex_document()

    assert r"\lstick{$q_{0}$} & \gate[" in latex_code
    assert r"& \meter[" in latex_code
    assert "\\begin{quantikz}" in latex_code
    assert "\\end{quantikz}" in latex_code
    assert DEFAULT_PREAMBLE_TEMPLATE.strip() in latex_code.strip()


def test_parameter_display():
    """Test that gate parameters are correctly displayed or hidden."""

    q_param = cirq.LineQubit(0)
    alpha = sympy.Symbol("\\alpha")  # Parameter symbol
    beta = sympy.Symbol("\\beta")  # Parameter symbol
    param_circuit = cirq.Circuit(
        cirq.H(q_param),
        cirq.rz(alpha).on(q_param),  # Parameterized gate
        cirq.X(q_param),
        cirq.X(q_param) ** sympy.Symbol("a_2"),
        cirq.X(q_param) ** 2.0,
        cirq.X(q_param) ** sympy.N(1.5),
        cirq.X(q_param) ** sympy.N(3.0),
        cirq.Y(q_param) ** 0.25,  # Parameterized exponent
        cirq.Y(q_param) ** alpha,  # Formula exponent
        cirq.X(q_param),  # Parameterized exponent
        cirq.rx(beta).on(q_param),  # Parameterized gate
        cirq.H(q_param),
        cirq.CZPowGate(exponent=0.25).on(q_param, cirq.q(2)),
        cirq.measure(q_param, key="result"),
    )
    # Test with show_parameters=True (default)
    converter_show_params = CircuitToQuantikz(param_circuit, show_parameters=True)
    latex_show_params = converter_show_params.generate_latex_document()
    assert r"R_{Z}(\alpha)" in latex_show_params
    assert r"Y^{0.25}" in latex_show_params
    assert r"H" in latex_show_params
    # Test with show_parameters=False
    converter_show_params = CircuitToQuantikz(param_circuit, show_parameters=False)
    latex_show_params = converter_show_params.generate_latex_document()
    assert r"R_{Z}(\alpha)" not in latex_show_params
    assert r"Y^{0.25}" not in latex_show_params
    assert r"H" in latex_show_params
    # Test with folding
    converter_show_params = CircuitToQuantikz(param_circuit, show_parameters=True, fold_at=5)
    latex_show_params = converter_show_params.generate_latex_document()
    assert r"R_{Z}(\alpha)" in latex_show_params
    assert r"Y^{0.25}" in latex_show_params
    assert r"H" in latex_show_params

    # Test with folding at boundary
    converter_show_params = CircuitToQuantikz(param_circuit, show_parameters=True, fold_at=1)
    latex_show_params = converter_show_params.generate_latex_document()
    assert r"R_{Z}(\alpha)" in latex_show_params
    assert r"Y^{0.25}" in latex_show_params
    assert r"H" in latex_show_params


def test_custom_gate_name_map():
    """Test custom gate name mapping."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.X(q))
    custom_map = {"H": "Hadamard"}
    converter = CircuitToQuantikz(circuit, gate_name_map=custom_map)
    latex_code = converter.generate_latex_document()

    assert r"Hadamard}" in latex_code
    assert r"{H}" not in latex_code  # Ensure original H is not there


def test_wire_labels():
    """Test different wire labeling options."""
    q0, q1, q2 = cirq.NamedQubit('alice'), cirq.LineQubit(10), cirq.GridQubit(4, 3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.X(q1), cirq.Z(q2))

    # 'q' labels
    converter_q = CircuitToQuantikz(circuit, wire_labels="q")
    latex_q = converter_q.generate_latex_document()
    assert r"\lstick{$q_{0}$}" in latex_q
    assert r"\lstick{$q_{1}$}" in latex_q
    assert r"\lstick{$q_{2}$}" in latex_q

    # 'index' labels
    converter_idx = CircuitToQuantikz(circuit, wire_labels="index")
    latex_idx = converter_idx.generate_latex_document()
    assert r"\lstick{$0$}" in latex_idx
    assert r"\lstick{$1$}" in latex_idx
    assert r"\lstick{$2$}" in latex_idx

    # 'qid' labels
    converter_q = CircuitToQuantikz(circuit, wire_labels="qid")
    latex_q = converter_q.generate_latex_document()
    assert r"\lstick{$alice$}" in latex_q
    assert r"\lstick{$q(10)$}" in latex_q
    assert r"\lstick{$q(4, 3)$}" in latex_q


def test_custom_preamble_and_postamble():
    """Test custom preamble and postamble injection."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    custom_preamble_text = r"\usepackage{mycustompackage}"
    custom_postamble_text = r"\end{tikzpicture}"

    converter = CircuitToQuantikz(
        circuit, custom_preamble=custom_preamble_text, custom_postamble=custom_postamble_text
    )
    latex_code = converter.generate_latex_document()

    assert custom_preamble_text in latex_code
    assert custom_postamble_text in latex_code
    assert "% --- Custom Preamble Injection Point ---" in latex_code
    assert "% --- Custom Postamble Start ---" in latex_code


def test_quantikz_options():
    """Test global quantikz options."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    options = "column sep=1em, row sep=0.5em"
    converter = CircuitToQuantikz(circuit, quantikz_options=options)
    latex_code = converter.generate_latex_document()

    assert f"\\begin{{quantikz}}[{options}]" in latex_code


def test_float_precision_exponents():
    """Test formatting of floating-point exponents."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q) ** 0.12345, cirq.Y(q) ** 0.5)
    converter = CircuitToQuantikz(circuit, float_precision_exps=3)
    latex_code = converter.generate_latex_document()
    assert r"X^{0.123}" in latex_code
    assert r"Y^{0.5}" in latex_code  # Should still be 0.5, not 0.500

    converter_int_exp = CircuitToQuantikz(circuit, float_precision_exps=0)
    latex_int_exp = converter_int_exp.generate_latex_document()
    assert r"X^{0.0}" in latex_int_exp  # 0.12345 rounded to 0
    assert r"Y^{0.0}" in latex_int_exp  # 0.5 is still 0.5 if not integer


def test_qubit_order():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.X.on_each(*qubits))
    qubit_order = cirq.QubitOrder.explicit([qubits[3], qubits[2], qubits[1], qubits[0]])
    converter = CircuitToQuantikz(circuit, qubit_order=qubit_order)
    latex_code = converter.generate_latex_document()
    q3 = latex_code.find("q(3)")
    q2 = latex_code.find("q(2)")
    q1 = latex_code.find("q(1)")
    q0 = latex_code.find("q(0)")
    assert q3 != -1
    assert q2 != -1
    assert q1 != -1
    assert q0 != -1
    assert q3 < q2
    assert q2 < q1
    assert q1 < q0


@pytest.mark.parametrize("show_parameters", [True, False])
def test_custom_gate(show_parameters) -> None:
    class CustomGate(cirq.Gate):
        def __init__(self, exponent: float | int):
            self.exponent = exponent

        def _num_qubits_(self):
            return 1

        def __str__(self):
            return f"Custom_Gate()**{self.exponent}"

        def _unitary_(self):
            raise NotImplementedError()

    circuit = cirq.Circuit(
        CustomGate(1.0).on(cirq.q(0)), CustomGate(1).on(cirq.q(0)), CustomGate(1.5).on(cirq.q(0))
    )
    converter = CircuitToQuantikz(circuit, show_parameters=show_parameters)
    latex_code = converter.generate_latex_document()
    assert "Custom" in latex_code


def test_misc_gates() -> None:
    """Tests gates that have special handling."""
    circuit = cirq.Circuit(cirq.global_phase_operation(-1), cirq.X(cirq.q(0)) ** 1.5)
    converter = CircuitToQuantikz(circuit)
    latex_code = converter.generate_latex_document()
    assert latex_code


def test_classical_control() -> None:
    circuit = cirq.Circuit(
        cirq.measure(cirq.q(0), key='a'), cirq.X(cirq.q(1)).with_classical_controls('a')
    )
    converter = CircuitToQuantikz(circuit)
    latex_code = converter.generate_latex_document()
    assert "\\vcw" in latex_code
