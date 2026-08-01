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

import pathlib
import subprocess
from unittest import mock

import numpy as np

import cirq
from cirq.contrib.quantikz.circuit_to_latex_render import render_circuit


def test_render_circuit(tmp_path: pathlib.Path) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.rx(0.25 * np.pi).on(q1),
        cirq.measure(q0, q1, key='result'),
    )
    # Render and display in Jupyter (if available), also save to a file
    img_or_path = render_circuit(
        circuit,
        output_png_path=tmp_path / "my_circuit.png",
        output_tex_path=tmp_path / "my_circuit.tex",
        output_pdf_path=tmp_path / "my_circuit.pdf",
        fold_at=2,
        debug=True,
        wire_labels="qid",
    )
    assert img_or_path is not None
    assert (tmp_path / "my_circuit.png").is_file()
    assert (tmp_path / "my_circuit.tex").is_file()
    assert (tmp_path / "my_circuit.pdf").is_file()


def test_render_circuit_disables_pdflatex_shell_escape() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q))
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as mock_run:
        render_circuit(circuit)
    pdflatex_calls = [call for call in mock_run.call_args_list if "pdflatex" in call.args[0][0]]
    assert pdflatex_calls
    for call in pdflatex_calls:
        assert "-no-shell-escape" in call.args[0]
        assert "-shell-escape" not in call.args[0]
