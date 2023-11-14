# Copyright 2023 The Cirq Developers
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

from pathlib import Path
from typing import Iterable

import cirq
import cirq.contrib.svg.svg as ccsvg
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import nbformat
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, get_named_qubits, merge_qubits
from nbconvert.preprocessors import ExecutePreprocessor


def display_gate_and_compilation(g: cq_testing.GateHelper, vertical=False, include_costs=True):
    """Use ipywidgets to display SVG circuits for a `GateHelper` next to each other.

    Args:
        g: The `GateHelper` to draw
        vertical: If true, lay-out the original gate and its decomposition vertically
            rather than side-by-side.
        include_costs: If true, each operation is annotated with it's T-complexity cost.
    """
    out1 = ipywidgets.Output()
    out2 = ipywidgets.Output()
    if vertical:
        box = ipywidgets.VBox([out1, out2])
    else:
        box = ipywidgets.HBox([out1, out2])

    out1.append_display_data(svg_circuit(g.circuit, registers=g.r, include_costs=include_costs))
    out2.append_display_data(
        svg_circuit(
            cirq.Circuit(cirq.decompose_once(g.operation)),
            registers=g.r,
            include_costs=include_costs,
        )
    )

    IPython.display.display(box)


def circuit_with_costs(circuit: 'cirq.AbstractCircuit') -> 'cirq.AbstractCircuit':
    """Annotates each operation in the circuit with its T-complexity cost."""

    def _map_func(op: cirq.Operation, _):
        t_cost = t_complexity_protocol.t_complexity(op)
        return op.with_tags(f't:{t_cost.t:g},r:{t_cost.rotations:g}')

    return cirq.map_operations(circuit, map_func=_map_func)


def svg_circuit(
    circuit: 'cirq.AbstractCircuit',
    registers: Iterable[gate_with_registers.Register] = (),
    include_costs: bool = False,
):
    """Return an SVG object representing a circuit.

    Args:
        circuit: The circuit to draw.
        registers: Optional `Signature` object to order the qubits.
        include_costs: If true, each operation is annotated with it's T-complexity cost.

    Raises:
        ValueError: If `circuit` is empty.
    """
    if len(circuit) == 0:
        raise ValueError("Circuit is empty.")

    if registers:
        qubit_order = cirq.QubitOrder.explicit(
            merge_qubits(registers, **get_named_qubits(registers)), fallback=cirq.QubitOrder.DEFAULT
        )
    else:
        qubit_order = cirq.QubitOrder.DEFAULT

    if include_costs:
        circuit = circuit_with_costs(circuit)

    tdd = circuit.to_text_diagram_drawer(transpose=False, qubit_order=qubit_order)
    if len(tdd.horizontal_lines) == 0:
        raise ValueError("No non-empty moments.")
    return IPython.display.SVG(ccsvg.tdd_to_svg(tdd))


def execute_notebook(name: str):
    """Execute a jupyter notebook in the caller's directory.

    Args:
        name: The name of the notebook without extension.
    """
    import traceback

    # Assumes that the notebook is in the same path from where the function was called,
    # which may be different from `__file__`.
    notebook_path = Path(traceback.extract_stack()[-2].filename).parent / f"{name}.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)
