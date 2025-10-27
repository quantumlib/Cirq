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

r"""Converts Cirq circuits to Quantikz LaTeX (using modern quantikz syntax).

This module provides a class, `CircuitToQuantikz`, to translate `cirq.Circuit`
objects into LaTeX code using the `quantikz` package. It aims to offer
flexible customization for gate styles, wire labels, and circuit folding.

Example:
    >>> import cirq
    >>> from cirq.contrib.quantikz import CircuitToQuantikz
    >>> q0, q1 = cirq.LineQubit.range(2)
    >>> circuit = cirq.Circuit(
    ...     cirq.H(q0),
    ...     cirq.CNOT(q0, q1),
    ...     cirq.measure(q0, key='m0'),
    ...     cirq.Rx(rads=0.5).on(q1)
    ... )
    >>> converter = CircuitToQuantikz(circuit, fold_at=2)
    >>> latex_code = converter.generate_latex_document()
    >>> print(latex_code)
    \documentclass[preview, border=2pt]{standalone}
    % Core drawing packages
    \usepackage{tikz}
    \usetikzlibrary{quantikz} % Loads the quantikz library (latest installed version)
    % Optional useful TikZ libraries
    \usetikzlibrary{fit, arrows.meta, decorations.pathreplacing, calligraphy}
    % Font encoding and common math packages
    \usepackage[T1]{fontenc}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    % --- Custom Preamble Injection Point ---
    % --- End Custom Preamble ---
    \begin{document}
    \begin{quantikz}
    \lstick{$q(0)$} & \gate[style={fill=yellow!20}]{H} & \qw & \rstick{$q(0)$} \\
    \lstick{$q(1)$} & \qw & \qw & \rstick{$q(1)$}
    \end{quantikz}
    <BLANKLINE>
    \vspace{1em}
    <BLANKLINE>
    \begin{quantikz}
    \lstick{$q(0)$} & \ctrl{1} & \meter[style={fill=gray!20}]{m0} & \qw & \rstick{$q(0)$} \\
    \lstick{$q(1)$} & \targ{} & \gate[style={fill=green!20}]{R_{X}(0.159\pi)} & \qw & \rstick{$q(1)$}
    \end{quantikz}
    \end{document}
"""  # noqa: E501

from __future__ import annotations

import math
import warnings
from typing import Any

import sympy

from cirq import circuits, ops, protocols, value

DEFAULT_PREAMBLE_TEMPLATE = r"""
\documentclass[preview, border=2pt]{standalone}
% Core drawing packages
\usepackage{tikz}
\usetikzlibrary{quantikz} % Loads the quantikz library (latest installed version)
% Optional useful TikZ libraries
\usetikzlibrary{fit, arrows.meta, decorations.pathreplacing, calligraphy}
% Font encoding and common math packages
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
""".lstrip()

# =============================================================================
# Default Style Definitions
# =============================================================================
_Pauli_gate_style = r"style={fill=blue!20}"
_green_gate_style = r"style={fill=green!20}"
_yellow_gate_style = r"style={fill=yellow!20}"  # For H
_orange_gate_style = r"style={fill=orange!20}"  # For FSim, iSwap, etc.
_gray_gate_style = r"style={fill=gray!20}"  # For Measure
_noisy_channel_style = r"style={fill=red!20}"

GATE_STYLES_COLORFUL = {
    "H": _yellow_gate_style,
    "_PauliX": _Pauli_gate_style,  # ops.X(q_param)
    "_PauliY": _Pauli_gate_style,  # ops.Y(q_param)
    "_PauliZ": _Pauli_gate_style,  # ops.Z(q_param)
    "X": _Pauli_gate_style,  # For XPowGate(exponent=1)
    "Y": _Pauli_gate_style,  # For YPowGate(exponent=1)
    "Z": _Pauli_gate_style,  # For ZPowGate(exponent=1)
    "X_pow": _green_gate_style,  # For XPowGate(exponent!=1)
    "Y_pow": _green_gate_style,  # For YPowGate(exponent!=1)
    "Z_pow": _green_gate_style,  # For ZPowGate(exponent!=1)
    "H_pow": _green_gate_style,  # For HPowGate(exponent!=1)
    "Rx": _green_gate_style,
    "Ry": _green_gate_style,
    "Rz": _green_gate_style,
    "PhasedXZ": _green_gate_style,
    "FSimGate": _orange_gate_style,
    "FSim": _orange_gate_style,  # Alias for FSimGate
    "iSwap": _orange_gate_style,  # For ISwapPowGate(exponent=1)
    "iSwap_pow": _orange_gate_style,  # For ISwapPowGate(exponent!=1)
    "CZ_pow": _orange_gate_style,  # For CZPowGate(exponent!=1)
    "CX_pow": _orange_gate_style,  # For CNotPowGate(exponent!=1)
    "CXideal": "",  # No fill for \ctrl \targ, let quantikz draw default
    "CZideal": "",  # No fill for \ctrl \control
    "Swapideal": "",  # No fill for \swap \targX
    "Measure": _gray_gate_style,
    "DepolarizingChannel": _noisy_channel_style,
    "BitFlipChannel": _noisy_channel_style,
    "ThermalChannel": _noisy_channel_style,
}


# Initialize gate maps globally as recommended
_SIMPLE_GATE_MAP: dict[type[ops.Gate], str] = {ops.MeasurementGate: "Measure"}
_EXPONENT_GATE_MAP: dict[type[ops.Gate], str] = {
    ops.XPowGate: "X",
    ops.YPowGate: "Y",
    ops.ZPowGate: "Z",
    ops.HPowGate: "H",
    ops.CNotPowGate: "CX",
    ops.CZPowGate: "CZ",
    ops.SwapPowGate: "Swap",
    ops.ISwapPowGate: "iSwap",
}
_GATE_NAME_MAP = {
    "Rx": r"R_{X}",
    "Ry": r"R_{Y}",
    "Rz": r"R_{Z}",
    "FSim": r"\mathrm{fSim}",
    "PhasedXZ": r"\Phi",
    "CZ": r"\mathrm{CZ}",
    "CX": r"\mathrm{CX}",
    "iSwap": r"i\mathrm{SWAP}",
}
_PARAMETERIZED_GATE_BASE_NAMES: dict[type[ops.Gate], str] = {
    ops.Rx: "Rx",
    ops.Ry: "Ry",
    ops.Rz: "Rz",
    ops.PhasedXZGate: "PhasedXZ",
    ops.FSimGate: "FSim",
}


# =============================================================================
# Cirq to Quantikz Conversion Class
# =============================================================================
class CircuitToQuantikz:
    r"""Converts a Cirq Circuit object to a Quantikz LaTeX string.

    This class facilitates the conversion of a `cirq.Circuit` into a LaTeX
    representation using the `quantikz` package. It handles various gate types,
    qubit mapping, and provides options for customizing the output, such as
    gate styling, circuit folding, and parameter display.

    Args:
        circuit: The `cirq.Circuit` object to be converted.
        gate_styles: An optional dictionary mapping gate names (strings) to
            Quantikz style options (strings). These styles are applied to
            the generated gates. If `None`, `GATE_STYLES_COLORFUL` is used.
        quantikz_options: An optional string of global options to pass to the
            `quantikz` environment (e.g., `"[row sep=0.5em]"`).
        fold_at: An optional integer specifying the number of moments after
            which the circuit should be folded into a new line in the LaTeX
            output. If `None`, the circuit is not folded.
        custom_preamble: An optional string containing custom LaTeX code to be
            inserted into the document's preamble.
        custom_postamble: An optional string containing custom LaTeX code to be
            inserted just before `\end{document}`.
        wire_labels: A string specifying how qubit wire labels should be
            rendered.
            - `"q"`: Labels as $q_0, q_1, \dots$
            - `"index"`: Labels as $0, 1, \dots$
            - `"qid"`: Labels as the string representation of the `cirq.Qid`
            - Any other value defaults to `"qid"`.
        show_parameters: A boolean indicating whether gate parameters (e.g.,
            exponents for `XPowGate`, angles for `Rx`) should be displayed
            in the gate labels.
        gate_name_map: An optional dictionary mapping Cirq gate names (strings)
            to custom LaTeX strings for rendering. This allows renaming gates
            in the output.
        float_precision_exps: An integer specifying the number of decimal
            places for formatting floating-point exponents.
        float_precision_angles: An integer specifying the number of decimal
            places for formatting floating-point angles. (Note: Not fully
            implemented in current version for all angle types).
        qubit_order:  Determines how qubits are ordered in the diagram.

    Raises:
        ValueError: If the input `circuit` is empty or contains no qubits.
    """

    def __init__(
        self,
        circuit: circuits.Circuit,
        *,
        gate_styles: dict[str, str] | None = None,
        quantikz_options: str | None = None,
        fold_at: int | None = None,
        custom_preamble: str = "",
        custom_postamble: str = "",
        wire_labels: str = "qid",
        show_parameters: bool = True,
        gate_name_map: dict[str, str] | None = None,
        float_precision_exps: int = 2,
        float_precision_angles: int = 2,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ):
        if not circuit:
            raise ValueError("Input circuit cannot be empty.")
        self.circuit = circuit
        self.gate_styles = gate_styles if gate_styles is not None else GATE_STYLES_COLORFUL.copy()
        self.quantikz_options = quantikz_options or ""
        self.fold_at = fold_at
        self.custom_preamble = custom_preamble
        self.custom_postamble = custom_postamble
        self.wire_labels = wire_labels
        self.show_parameters = show_parameters
        self.current_gate_name_map = _GATE_NAME_MAP.copy()
        if gate_name_map:
            self.current_gate_name_map.update(gate_name_map)
        self.sorted_qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.circuit.all_qubits()
        )
        if not self.sorted_qubits:
            raise ValueError("Circuit contains no qubits.")
        self.qubit_to_index = self._map_qubits_to_indices()
        self.key_to_index = self._map_keys_to_indices()
        self.num_qubits = len(self.sorted_qubits)
        self.float_precision_exps = float_precision_exps
        self.float_precision_angles = float_precision_angles

    def _map_qubits_to_indices(self) -> dict[ops.Qid, int]:
        """Creates a mapping from `cirq.Qid` objects to their corresponding
        integer indices based on the sorted qubit order.

        Returns:
            A dictionary where keys are `cirq.Qid` objects and values are their
            zero-based integer indices.
        """
        return {q: i for i, q in enumerate(self.sorted_qubits)}

    def _map_keys_to_indices(self) -> dict[str, list[int]]:
        """Maps measurement keys to qubit indices.

        Used by classically controlled operations to map keys
        to qubit wires.
        """
        key_map: dict[str, list[int]] = {}
        for op in self.circuit.all_operations():
            if isinstance(op.gate, ops.MeasurementGate):
                key_map[op.gate.key] = [self.qubit_to_index[q] for q in op.qubits]
        return key_map

    def _escape_string(self, label) -> str:
        """Escape labels for latex."""
        label = label.replace("π", r"\pi")
        if "_" in label and "\\" not in label:
            label = label.replace("_", r"\_")
        return label

    def _get_wire_label(self, qubit: ops.Qid, index: int) -> str:
        r"""Generates the LaTeX string for a qubit wire label.

        Args:
            qubit: The `cirq.Qid` object for which to generate the label.
            index: The integer index of the qubit.

        Returns:
            A string formatted as a LaTeX math-mode label (e.g., "$q_0$", "$3$",
            or "$q_{qubit\_name}$").
        """
        lbl = (
            f"q_{{{index}}}"
            if self.wire_labels == "q"
            else (
                str(index) if self.wire_labels == "index" else str(self._escape_string(str(qubit)))
            )
        )
        return f"${lbl}$"

    def _format_exponent_for_display(self, exponent: Any) -> str:
        """Formats a gate exponent for display in LaTeX.

        Handles floats, integers, and `sympy.Basic` expressions, converting them
        to a string representation suitable for LaTeX, including proper
        handling of numerical precision and symbolic constants like pi.

        Args:
            exponent: The exponent value, which can be a float, int, or
                `sympy.Basic` object.

        Returns:
            A string representing the formatted exponent, ready for LaTeX
            insertion.
        """
        exp_str: str
        # Dynamically create the format string based on self.float_precision_exps
        float_format_string = f".{self.float_precision_exps}f"

        if isinstance(exponent, float):
            # If the float is an integer value (e.g., 2.0), display as integer string ("2")
            if exponent.is_integer():
                exp_str = str(int(exponent))
            else:
                # Format to the specified precision for rounding
                rounded_str = format(exponent, float_format_string)
                # Convert back to float and then to string to remove unnecessary trailing zeros
                # e.g., if precision is 2, 0.5 -> "0.50" -> 0.5 -> "0.5"
                # e.g., if precision is 2, 0.318 -> "0.32" -> 0.32 -> "0.32"
                exp_str = str(float(rounded_str))
        # Check for sympy.Basic, assuming sympy is imported if this path is taken
        elif isinstance(exponent, sympy.Basic):
            s_exponent = str(exponent)
            # Heuristic: check for letters to identify symbolic expressions
            is_symbolic_or_special = any(
                char.isalpha()
                for char in s_exponent
                if char.lower() not in ["e"]  # Exclude 'e' for scientific notation
            )
            if not is_symbolic_or_special:  # If it looks like a number
                try:
                    py_float = float(sympy.N(exponent))
                    # If the sympy evaluated float is an integer value
                    if py_float.is_integer():
                        exp_str = str(int(py_float))
                    else:
                        # Format to specified precision for rounding
                        rounded_str = format(py_float, float_format_string)
                        # Convert back to float then to string to remove unnecessary trailing zeros
                        exp_str = str(float(rounded_str))
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    sympy.SympifyError,
                ):  # pragma: no cover
                    # Fallback to Sympy's string representation if conversion fails
                    exp_str = s_exponent
            else:  # Symbolic expression
                exp_str = s_exponent
        else:  # For other types (int, strings not sympy objects)
            exp_str = str(exponent)

        return self._escape_string(exp_str)

    def _get_gate_name(self, gate: ops.Gate) -> str:
        """Determines the appropriate LaTeX string for a given Cirq gate.

        This method attempts to derive a suitable LaTeX name for the gate,
        considering its type, whether it's a power gate, and if parameters
        should be displayed. It uses internal mappings and `cirq.circuit_diagram_info`.

        Args:
            gate: The `cirq.Gate` object to name.

        Returns:
            A string representing the LaTeX name of the gate (e.g., "H",
            "Rx(0.5)", "CZ").
        """
        gate_type = type(gate)
        if (simple_name := _SIMPLE_GATE_MAP.get(gate_type)) is not None:
            return simple_name

        base_key = _EXPONENT_GATE_MAP.get(gate_type)
        if base_key is not None and hasattr(gate, "exponent") and gate.exponent == 1:
            return self.current_gate_name_map.get(base_key, base_key)

        if (param_base_key := _PARAMETERIZED_GATE_BASE_NAMES.get(gate_type)) is not None:
            mapped_name = self.current_gate_name_map.get(param_base_key, param_base_key)
            if not self.show_parameters:
                return mapped_name
            # Use protocols directly
            info = protocols.circuit_diagram_info(gate, default=NotImplemented)
            if info is not NotImplemented and info.wire_symbols:
                s_diag = info.wire_symbols[0]
                if (op_idx := s_diag.find("(")) != -1 and (cp_idx := s_diag.rfind(")")) > op_idx:
                    return (
                        f"{mapped_name}"
                        f"({self._format_exponent_for_display(s_diag[op_idx+1:cp_idx])})"
                    )
            if hasattr(gate, "exponent") and not math.isclose(
                gate.exponent, 1.0
            ):  # pragma: no cover
                return f"{mapped_name}({self._format_exponent_for_display(gate.exponent)})"
            return mapped_name  # pragma: no cover

        try:
            # Use protocols directly
            info = protocols.circuit_diagram_info(gate, default=NotImplemented)
            if info is not NotImplemented and info.wire_symbols:
                name_cand = info.wire_symbols[0]
                if not self.show_parameters:
                    base_part = name_cand.split("^")[0].split("**")[0].split("(")[0].strip()
                    if isinstance(gate, ops.CZPowGate) and base_part == "@":
                        base_part = "CZ"
                    mapped_base = self.current_gate_name_map.get(base_part, base_part)
                    return self._format_exponent_for_display(mapped_base)

                if (
                    hasattr(gate, "exponent")
                    and not (isinstance(gate.exponent, float) and math.isclose(gate.exponent, 1.0))
                    and isinstance(gate, tuple(_EXPONENT_GATE_MAP.keys()))
                ):
                    has_exp_in_cand = ("^" in name_cand) or ("**" in name_cand)
                    if not has_exp_in_cand and base_key:
                        recon_base = self.current_gate_name_map.get(base_key, base_key)
                        needs_recon = (name_cand == base_key) or (
                            isinstance(gate, ops.CZPowGate) and name_cand == "@"
                        )
                        if needs_recon:
                            name_cand = (
                                f"{recon_base}^"
                                f"{{{self._format_exponent_for_display(gate.exponent)}}}"
                            )

                fmt_name = self._escape_string(name_cand)
                parts = fmt_name.split("**", 1)
                if len(parts) == 2:  # pragma: no cover
                    fmt_name = f"{parts[0]}^{{{self._format_exponent_for_display(parts[1])}}}"
                return fmt_name
        except (ValueError, AttributeError, IndexError):  # pragma: no cover
            # Fallback to default string representation if diagram info parsing fails.
            pass

        name_fb = str(gate)
        if name_fb.endswith("**1.0"):
            name_fb = name_fb[:-5]
        if name_fb.endswith("**1"):
            name_fb = name_fb[:-3]
        if name_fb.endswith("()"):
            name_fb = name_fb[:-2]
        if name_fb.endswith("Gate"):
            name_fb = name_fb[:-4]
        if not self.show_parameters:
            base_fb = name_fb.split("**")[0].split("(")[0].strip()
            fb_key = _EXPONENT_GATE_MAP.get(gate_type, base_fb)
            mapped_fb = self.current_gate_name_map.get(fb_key, fb_key)
            return self._format_exponent_for_display(mapped_fb)
        if "**" in name_fb:
            parts = name_fb.split("**", 1)
            if len(parts) == 2:
                fb_key = _EXPONENT_GATE_MAP.get(gate_type, parts[0])
                base_str_fb = self.current_gate_name_map.get(fb_key, parts[0])
                name_fb = f"{base_str_fb}^{{{self._format_exponent_for_display(parts[1])}}}"
        return self._escape_string(name_fb)

    def _get_quantikz_options_string(self) -> str:
        return f"[{self.quantikz_options}]" if self.quantikz_options else ""

    def _render_operation(self, op: ops.Operation) -> dict[int, str]:
        """Renders a single Cirq operation into its Quantikz LaTeX string representation.

        Handles various gate types, including single-qubit gates, multi-qubit gates,
        measurement gates, and special control/target gates (CNOT, CZ, SWAP).
        Applies appropriate styles and labels based on the gate type and
        `CircuitToQuantikz` instance settings.

        Args:
            op: The `cirq.Operation` object to render.

        Returns:
            A dictionary mapping qubit indices to their corresponding LaTeX strings
            for the current moment.
        """
        output, q_indices = {}, sorted([self.qubit_to_index[q] for q in op.qubits])
        gate = op.gate
        if isinstance(op, ops.ClassicallyControlledOperation):
            gate = op.without_classical_controls().gate
        if gate is None:  # pragma: no cover
            raise ValueError(f'Only GateOperations are supported {op}')
        gate_name_render = self._get_gate_name(gate)

        gate_type = type(gate)
        style_key = gate_type.__name__  # Default style key

        # Determine style key based on gate type and properties
        if isinstance(gate, ops.CNotPowGate) and gate.exponent == 1:
            style_key = "CXideal"
        elif isinstance(gate, ops.CZPowGate) and gate.exponent == 1:
            style_key = "CZideal"
        elif isinstance(gate, ops.SwapPowGate) and gate.exponent == 1:
            style_key = "Swapideal"
        elif isinstance(gate, ops.MeasurementGate):
            style_key = "Measure"
        elif (param_base_name := _PARAMETERIZED_GATE_BASE_NAMES.get(gate_type)) is not None:
            style_key = param_base_name
        elif (base_key_for_pow := _EXPONENT_GATE_MAP.get(gate_type)) is not None:
            if getattr(gate, "exponent", 1) == 1:
                style_key = base_key_for_pow
            else:
                style_key = f"{base_key_for_pow}_pow"

        style_opts_str = self.gate_styles.get(style_key, "")
        final_style_tikz = f"[{style_opts_str}]" if style_opts_str else ""

        # Apply special Quantikz commands for specific gate types
        if isinstance(gate, ops.MeasurementGate):
            lbl = gate.key.replace("_", r"\_") if gate.key else ""
            for idx, q1 in enumerate(q_indices):
                if idx == 0:
                    output[q1] = f"\\meter{final_style_tikz}{{{lbl}}}"
                else:
                    q0 = q_indices[idx - 1]
                    output[q1] = f"\\meter{final_style_tikz}{{}} \\vqw{{{q0-q1}}}"
        elif isinstance(gate, ops.CNotPowGate) and gate.exponent == 1:
            c, t = (
                (self.qubit_to_index[op.qubits[0]], self.qubit_to_index[op.qubits[1]])
                if len(op.qubits) == 2
                else (q_indices[0], q_indices[0])
            )
            output[c] = f"\\ctrl{final_style_tikz}{{{t-c}}}"
            output[t] = f"\\targ{final_style_tikz}{{}}"
        elif isinstance(gate, ops.CZPowGate) and gate.exponent == 1:
            i1, i2 = (
                (q_indices[0], q_indices[1])
                if len(q_indices) >= 2
                else (q_indices[0], q_indices[0])
            )
            output[i1] = f"\\ctrl{final_style_tikz}{{{i2-i1}}}"
            output[i2] = f"\\control{final_style_tikz}{{}}"
        elif isinstance(gate, ops.SwapPowGate) and gate.exponent == 1:
            i1, i2 = (
                (q_indices[0], q_indices[1])
                if len(q_indices) >= 2
                else (q_indices[0], q_indices[0])
            )
            output[i1] = f"\\swap{final_style_tikz}{{{i2-i1}}}"
            output[i2] = f"\\targX{final_style_tikz}{{}}"
        # Handle generic \gate command for single and multi-qubit gates
        elif len(q_indices) == 1:
            output[q_indices[0]] = f"\\gate{final_style_tikz}{{{gate_name_render}}}"
        else:  # Multi-qubit gate
            combined_opts = f"wires={q_indices[-1]-q_indices[0]+1}"
            if style_opts_str:
                combined_opts = f"{combined_opts}, {style_opts_str}"
            output[q_indices[0]] = f"\\gate[{combined_opts}]{{{gate_name_render}}}"
            for i in range(1, len(q_indices)):
                output[q_indices[i]] = "\\qw"
        if isinstance(op, ops.ClassicallyControlledOperation):
            q0 = q_indices[0]
            for key in op.classical_controls:
                if isinstance(key, value.KeyCondition):
                    for index in self.key_to_index[key.key.name]:
                        output[q0] += f" \\vcw{{{index-q0}}}"
                        output[index] = "\\ctrl{}"

        return output

    def _initial_active_chunk(self) -> list[list[str]]:
        """Add initial wire labels for the first chunk"""
        return [
            [f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}"]
            for i in range(self.num_qubits)
        ]

    def _generate_latex_body(self) -> str:
        """Generates the main LaTeX body for the circuit diagram.

        Iterates through the circuit's moments, renders each operation, and
        arranges them into Quantikz environments. Supports circuit folding
        into multiple rows if `fold_at` is specified.
        Handles qubit wire labels and ensures correct LaTeX syntax.
        """
        chunks = []
        active_chunk = self._initial_active_chunk()

        for m_idx, moment in enumerate(self.circuit):
            moment_out = ["\\qw"] * self.num_qubits

            # Add LaTeX for each operation in the moment
            spanned_qubits: set[int] = set()
            for op in moment:
                if not op.qubits:
                    warnings.warn(f"Op {op} no qubits.")
                    continue
                min_qubit = min(self.qubit_to_index[q] for q in op.qubits)
                max_qubit = max(self.qubit_to_index[q] for q in op.qubits)
                for i in range(min_qubit, max_qubit + 1):
                    if i in spanned_qubits:
                        # This overlaps another operation:
                        # Create a new column.
                        for i in range(self.num_qubits):
                            active_chunk[i].append(moment_out[i])
                        moment_out = ["\\qw"] * self.num_qubits
                        spanned_qubits = set()
                for i in range(min_qubit, max_qubit + 1):
                    spanned_qubits.add(i)
                for q in op.qubits:
                    spanned_qubits.add(self.qubit_to_index[q])
                op_rnd = self._render_operation(op)
                for idx, tex in op_rnd.items():
                    moment_out[idx] = tex
            for i in range(self.num_qubits):
                active_chunk[i].append(moment_out[i])

            is_last_m = m_idx == len(self.circuit) - 1
            if self.fold_at and m_idx % self.fold_at == 0 and not is_last_m:
                for i in range(self.num_qubits):
                    lbl = self._get_wire_label(self.sorted_qubits[i], i)
                    active_chunk[i].extend(["\\qw", f"\\rstick{{{lbl}}}"])
                chunks.append(active_chunk)
                active_chunk = self._initial_active_chunk()

        for i in range(self.num_qubits):
            active_chunk[i].append("\\qw")
        if self.fold_at:
            for i in range(self.num_qubits):
                active_chunk[i].extend(
                    [f"\\rstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}"]
                )
        chunks.append(active_chunk)

        opts_str = self._get_quantikz_options_string()
        final_parts = []
        for chunk_data in chunks:
            lines = [f"\\begin{{quantikz}}{opts_str}"]
            for i in range(self.num_qubits):
                if i < len(chunk_data) and chunk_data[i]:
                    lines.append(" & ".join(chunk_data[i]) + " \\\\")

            if len(lines) > 1:
                for k_idx in range(len(lines) - 1, 0, -1):
                    stipped_line = lines[k_idx].strip()
                    if stipped_line:
                        if stipped_line != "\\\\":
                            if lines[k_idx].endswith(" \\\\"):
                                lines[k_idx] = lines[k_idx].rstrip()[:-3].rstrip()
                            break
                        elif k_idx == len(lines) - 1:  # pragma: no cover
                            lines[k_idx] = ""
            lines.append("\\end{quantikz}")
            final_parts.append("\n".join(filter(None, lines)))

        return "\n\n\\vspace{1em}\n\n".join(final_parts)

    def generate_latex_document(self, preamble_template: str | None = None) -> str:
        """Generates the complete LaTeX document string for the circuit.

        Combines the preamble, custom preamble, generated circuit body,
        and custom postamble into a single LaTeX document string.

        Args:
            preamble_template: An optional string to use as the base LaTeX
                preamble. If `None`, `DEFAULT_PREAMBLE_TEMPLATE` is used.

        Returns:
            A string containing the full LaTeX document, ready to be compiled.
        """
        preamble = preamble_template or DEFAULT_PREAMBLE_TEMPLATE
        doc_parts = [
            preamble.rstrip(),
            "% --- Custom Preamble Injection Point ---",
            *([self.custom_preamble.rstrip()] if self.custom_preamble else []),
            "% --- End Custom Preamble ---",
            "\\begin{document}",
            self._generate_latex_body(),
        ]
        if self.custom_postamble:
            doc_parts.append(
                f"% --- Custom Postamble Start ---\n"
                f"{self.custom_postamble.rstrip()}\n"
                f"% --- Custom Postamble End ---"
            )
        doc_parts.append("\\end{document}")
        return "\n".join(doc_parts)
