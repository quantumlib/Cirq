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

import pytest
import cirq
import sympy
import numpy as np

# Import the class directly for testing
from cirq.vis.circuit_to_latex_quantikz import CircuitToQuantikz, DEFAULT_PREAMBLE_TEMPLATE


def test_empty_circuit_raises_value_error():
    """Test that an empty circuit raises a ValueError."""
    empty_circuit = cirq.Circuit()
    with pytest.raises(ValueError, match="Input circuit cannot be empty."):
        CircuitToQuantikz(empty_circuit)


def test_basic_circuit_conversion():
    """Test a simple circuit conversion to LaTeX."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, key='m0'))
    converter = CircuitToQuantikz(circuit)
    latex_code = converter.generate_latex_document()
    # print(latex_code)

    assert r"\lstick{$q_{0}$} & \gate[" in latex_code
    assert "& \meter[" in latex_code
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
        cirq.Y(q_param) ** 0.25,  # Parameterized exponent
        cirq.X(q_param),  # Parameterized exponent
        cirq.rx(beta).on(q_param),  # Parameterized gate
        cirq.H(q_param),
        cirq.measure(q_param, key="result"),
    )
    print(param_circuit)
    # Test with show_parameters=True (default)
    converter_show_params = CircuitToQuantikz(param_circuit, show_parameters=True)
    latex_show_params = converter_show_params.generate_latex_document()
    print(latex_show_params)
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
    print(latex_code)

    assert r"Hadamard}" in latex_code
    assert r"{H}" not in latex_code  # Ensure original H is not there


def test_wire_labels():
    """Test different wire labeling options."""
    q0, q1 = cirq.NamedQubit('alice'), cirq.LineQubit(10)
    circuit = cirq.Circuit(cirq.H(q0), cirq.X(q1))

    # Default 'q' labels
    converter_q = CircuitToQuantikz(circuit, wire_labels="q")
    latex_q = converter_q.generate_latex_document()
    # print(latex_q)
    assert r"\lstick{$q_{0}$}" in latex_q
    assert r"\lstick{$q_{1}$}" in latex_q

    # 'index' labels
    converter_idx = CircuitToQuantikz(circuit, wire_labels="index")
    latex_idx = converter_idx.generate_latex_document()
    assert r"\lstick{$0$}" in latex_idx
    assert r"\lstick{$1$}" in latex_idx


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
    print(latex_code)
    assert r"X^{0.123}" in latex_code
    assert r"Y^{0.5}" in latex_code  # Should still be 0.5, not 0.500

    converter_int_exp = CircuitToQuantikz(circuit, float_precision_exps=0)
    latex_int_exp = converter_int_exp.generate_latex_document()
    # print(latex_int_exp)
    assert r"X^{0.0}" in latex_int_exp  # 0.12345 rounded to 0
    assert r"Y^{0.0}" in latex_int_exp  # 0.5 is still 0.5 if not integer
