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

# -*- coding: utf-8 -*-
r"""Converts Cirq circuits to Quantikz LaTeX (using modern quantikz syntax).

This module provides a class, `CircuitToQuantikz`, to translate `cirq.Circuit`
objects into LaTeX code using the `quantikz` package. It aims to offer
flexible customization for gate styles, wire labels, and circuit folding.

Example:
    >>> import cirq
    >>> from cirq.vis.circuit_to_latex_quantikz import CircuitToQuantikz
    >>> q0, q1 = cirq.LineQubit.range(2)
    >>> circuit = cirq.Circuit(
    ...     cirq.H(q0),
    ...     cirq.CNOT(q0, q1),
    ...     cirq.measure(q0, key='m0'),
    ...     cirq.Rx(rads=0.5).on(q1)
    ... )
    >>> converter = CircuitToQuantikz(circuit, fold_at=2)
    >>> latex_code = converter.generate_latex_document()
    >>> print(latex_code) # doctest: +SKIP
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
    % \usepackage{physics} % Removed
    % --- Custom Preamble Injection Point ---
    % --- End Custom Preamble ---
    \begin{document}
    \begin{quantikz}
    \lstick{$q_{0}$} & \gate[style={fill=yellow!20}]{H} & \ctrl{1} & \meter{$m0$} \\
    \lstick{$q_{1}$} & \qw & \targ{} & \qw
    \end{quantikz}

    \vspace{1em}

    \begin{quantikz}
    \lstick{$q_{0}$} & \qw & \rstick{$q_{0}$} \\
    \lstick{$q_{1}$} & \gate[style={fill=green!20}]{R_{X}(0.5)} & \rstick{$q_{1}$}
    \end{quantikz}
    \end{document}
"""

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import sympy

from cirq import circuits, devices, ops, protocols

__all__ = ["CircuitToQuantikz", "DEFAULT_PREAMBLE_TEMPLATE", "GATE_STYLES_COLORFUL1"]


# =============================================================================
# Default Preamble Template (physics.sty removed)
# =============================================================================
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
% \usepackage{physics} % Removed
"""

# =============================================================================
# Default Style Definitions
# =============================================================================
_Pauli_gate_style = r"style={fill=blue!20}"
_green_gate_style = r"style={fill=green!20}"
_yellow_gate_style = r"style={fill=yellow!20}"  # For H
_orange_gate_style = r"style={fill=orange!20}"  # For FSim, ISwap, etc.
_gray_gate_style = r"style={fill=gray!20}"  # For Measure
_noisy_channel_style = r"style={fill=red!20}"

GATE_STYLES_COLORFUL1 = {
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
    "ISwap": _orange_gate_style,  # For ISwapPowGate(exponent=1)
    "iSWAP_pow": _orange_gate_style,  # For ISwapPowGate(exponent!=1)
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
_SIMPLE_GATE_MAP: Dict[Type[ops.Gate], str] = {ops.MeasurementGate: "Measure"}
_EXPONENT_GATE_MAP: Dict[Type[ops.Gate], str] = {
    ops.XPowGate: "X",
    ops.YPowGate: "Y",
    ops.ZPowGate: "Z",
    ops.HPowGate: "H",
    ops.CNotPowGate: "CX",
    ops.CZPowGate: "CZ",
    ops.SwapPowGate: "Swap",
    ops.ISwapPowGate: "iSwap",
}
_PARAMETERIZED_GATE_BASE_NAMES: Dict[Type[ops.Gate], str] = {}
_param_gate_specs = [
    ("Rx", getattr(ops, "Rx", None)),
    ("Ry", getattr(ops, "Ry", None)),
    ("Rz", getattr(ops, "Rz", None)),
    ("PhasedXZ", getattr(ops, "PhasedXZGate", None)),
    ("FSim", getattr(ops, "FSimGate", None)),
]
if _param_gate_specs:
    for _name, _gate_cls in _param_gate_specs:
        if _gate_cls:
            _PARAMETERIZED_GATE_BASE_NAMES[_gate_cls] = _name


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
            the generated gates. If `None`, `GATE_STYLES_COLORFUL1` is used.
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
            - Any other value defaults to `"q"`.
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

    Raises:
        ValueError: If the input `circuit` is empty or contains no qubits.
    """

    GATE_NAME_MAP = {
        "Rx": r"R_{X}",
        "Ry": r"R_{Y}",
        "Rz": r"R_{Z}",
        "FSim": r"\mathrm{fSim}",
        "PhasedXZ": r"\Phi",
        "CZ": r"\mathrm{CZ}",
        "CX": r"\mathrm{CX}",
        "iSwap": r"i\mathrm{SWAP}",
    }

    def __init__(
        self,
        circuit: circuits.Circuit,
        *,
        gate_styles: Optional[Dict[str, str]] = None,
        quantikz_options: Optional[str] = None,
        fold_at: Optional[int] = None,
        custom_preamble: str = "",
        custom_postamble: str = "",
        wire_labels: str = "q",
        show_parameters: bool = True,
        gate_name_map: Optional[Dict[str, str]] = None,
        float_precision_exps: int = 2,
        float_precision_angles: int = 2,
    ):
        if not circuit:
            raise ValueError("Input circuit cannot be empty.")
        self.circuit = circuit
        self.gate_styles = gate_styles if gate_styles is not None else GATE_STYLES_COLORFUL1.copy()
        self.quantikz_options = quantikz_options or ""
        self.fold_at = fold_at
        self.custom_preamble = custom_preamble
        self.custom_postamble = custom_postamble
        self.wire_labels = wire_labels
        self.show_parameters = show_parameters
        self.current_gate_name_map = self.GATE_NAME_MAP.copy()
        if gate_name_map:
            self.current_gate_name_map.update(gate_name_map)
        self.sorted_qubits = self._get_sorted_qubits()
        if not self.sorted_qubits:
            raise ValueError("Circuit contains no qubits.")
        self.qubit_to_index = self._map_qubits_to_indices()
        self.num_qubits = len(self.sorted_qubits)
        self.float_precision_exps = float_precision_exps
        self.float_precision_angles = float_precision_angles

        # Gate maps are now global, no need to initialize here.
        self._SIMPLE_GATE_MAP = _SIMPLE_GATE_MAP
        self._EXPONENT_GATE_MAP = _EXPONENT_GATE_MAP
        self._PARAMETERIZED_GATE_BASE_NAMES = _PARAMETERIZED_GATE_BASE_NAMES

    def _get_sorted_qubits(self) -> List[ops.Qid]:
        """Determines and returns a sorted list of all unique qubits in the circuit.

        Returns:
            A list of `cirq.Qid` objects, sorted to ensure consistent qubit
            ordering in the LaTeX output.
        """
        qubits = set(q for moment in self.circuit for op in moment for q in op.qubits)
        return sorted(list(qubits))

    def _map_qubits_to_indices(self) -> Dict[ops.Qid, int]:
        """Creates a mapping from `cirq.Qid` objects to their corresponding
        integer indices based on the sorted qubit order.

        Returns:
            A dictionary where keys are `cirq.Qid` objects and values are their
            zero-based integer indices.
        """
        return {q: i for i, q in enumerate(self.sorted_qubits)}

    def _get_wire_label(self, qubit: ops.Qid, index: int) -> str:
        r"""Generates the LaTeX string for a qubit wire label.

        Args:
            qubit: The `cirq.Qid` object for which to generate the label.
            index: The integer index of the qubit.

        Returns:
            A string formatted as a LaTeX math-mode label (e.g., "$q_0$", "$3$",
            or "$q_{qubit\_name}$").
        """
        s = str(qubit).replace("_", r"\_").replace(" ", r"\,")
        lbl = (
            f"q_{{{index}}}"
            if self.wire_labels == "q"
            else (
                str(index)
                if self.wire_labels == "index"
                else s if self.wire_labels == "qid" else f"q_{{{index}}}"
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
                        # Convert back to float and then to string to remove unnecessary trailing zeros
                        exp_str = str(float(rounded_str))
                except (TypeError, ValueError, AttributeError, sympy.SympifyError):
                    # Fallback to Sympy's string representation if conversion fails
                    exp_str = s_exponent
            else:  # Symbolic expression
                exp_str = s_exponent
        else:  # For other types (int, strings not sympy objects)
            exp_str = str(exponent)

        # LaTeX replacements for pi
        exp_str = exp_str.replace("pi", r"\pi").replace("π", r"\pi")

        # Handle underscores: replace "_" with "\_" if not part of a LaTeX command
        if "_" in exp_str and "\\" not in exp_str:
            exp_str = exp_str.replace("_", r"\_")
        return exp_str

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
        if gate_type.__name__ == "ThermalChannel":
            return "\\Lambda_\\mathrm{th}"
        if (simple_name := self._SIMPLE_GATE_MAP.get(gate_type)) is not None:
            return simple_name

        base_key = self._EXPONENT_GATE_MAP.get(gate_type)
        if base_key is not None and hasattr(gate, "exponent") and gate.exponent == 1:
            return self.current_gate_name_map.get(base_key, base_key)

        if (param_base_key := self._PARAMETERIZED_GATE_BASE_NAMES.get(gate_type)) is not None:
            mapped_name = self.current_gate_name_map.get(param_base_key, param_base_key)
            if not self.show_parameters:
                return mapped_name
            try:
                # Use protocols directly
                info = protocols.circuit_diagram_info(gate, default=NotImplemented)
                if info is not NotImplemented and info.wire_symbols:
                    s_diag = info.wire_symbols[0]
                    if (op_idx := s_diag.find("(")) != -1 and (
                        cp_idx := s_diag.rfind(")")
                    ) > op_idx:
                        return f"{mapped_name}({self._format_exponent_for_display(s_diag[op_idx+1:cp_idx])})"
            except (ValueError, AttributeError, IndexError):
                # Fallback to default string representation if diagram info parsing fails.
                pass
            if hasattr(gate, "exponent") and not math.isclose(gate.exponent, 1.0):
                return f"{mapped_name}({self._format_exponent_for_display(gate.exponent)})"
            return mapped_name

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
                    and not math.isclose(gate.exponent, 1.0)
                    and isinstance(gate, tuple(self._EXPONENT_GATE_MAP.keys()))
                ):
                    has_exp_in_cand = ("^" in name_cand) or ("**" in name_cand)
                    if not has_exp_in_cand and base_key:
                        recon_base = self.current_gate_name_map.get(base_key, base_key)
                        needs_recon = (name_cand == base_key) or (
                            isinstance(gate, ops.CZPowGate) and name_cand == "@"
                        )
                        if needs_recon:
                            name_cand = f"{recon_base}^{{{self._format_exponent_for_display(gate.exponent)}}}"

                fmt_name = name_cand.replace("π", r"\pi")
                if "_" in fmt_name and "\\" not in fmt_name:
                    fmt_name = fmt_name.replace("_", r"\_")
                if "**" in fmt_name:
                    parts = fmt_name.split("**", 1)
                    if len(parts) == 2:
                        fmt_name = f"{parts[0]}^{{{self._format_exponent_for_display(parts[1])}}}"
                return fmt_name
        except (ValueError, AttributeError, IndexError):
            # Fallback to default string representation if diagram info parsing fails.
            pass

        name_fb = str(gate)
        if name_fb.endswith("Gate"):
            name_fb = name_fb[:-4]
        if name_fb.endswith("()"):
            name_fb = name_fb[:-2]
        if not self.show_parameters:
            base_fb = name_fb.split("**")[0].split("(")[0].strip()
            fb_key = self._EXPONENT_GATE_MAP.get(gate_type, base_fb)
            mapped_fb = self.current_gate_name_map.get(fb_key, fb_key)
            return self._format_exponent_for_display(mapped_fb)
        if name_fb.endswith("**1.0"):
            name_fb = name_fb[:-5]
        if name_fb.endswith("**1"):
            name_fb = name_fb[:-3]
        if "**" in name_fb:
            parts = name_fb.split("**", 1)
            if len(parts) == 2:
                fb_key = self._EXPONENT_GATE_MAP.get(gate_type, parts[0])
                base_str_fb = self.current_gate_name_map.get(fb_key, parts[0])
                name_fb = f"{base_str_fb}^{{{self._format_exponent_for_display(parts[1])}}}"
        name_fb = name_fb.replace("π", r"\pi")
        if "_" in name_fb and "\\" not in name_fb:
            name_fb = name_fb.replace("_", r"\_")
        return name_fb

    def _get_quantikz_options_string(self) -> str:
        return f"[{self.quantikz_options}]" if self.quantikz_options else ""

    def _render_operation(self, op: ops.Operation) -> Dict[int, str]:
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
        gate, gate_name_render = op.gate, self._get_gate_name(op.gate)

        gate_type = type(gate)
        style_key = gate_type.__name__  # Default style key

        # Determine style key based on gate type and properties
        if isinstance(gate, ops.CNotPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            style_key = "CXideal"
        elif isinstance(gate, ops.CZPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            style_key = "CZideal"
        elif isinstance(gate, ops.SwapPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            style_key = "Swapideal"
        elif isinstance(gate, ops.MeasurementGate):
            style_key = "Measure"
        elif (param_base_name := self._PARAMETERIZED_GATE_BASE_NAMES.get(gate_type)) is not None:
            style_key = param_base_name
        elif (base_key_for_pow := self._EXPONENT_GATE_MAP.get(gate_type)) is not None:
            if hasattr(gate, "exponent"):
                if gate.exponent == 1:
                    style_key = base_key_for_pow
                else:
                    style_key = {
                        "X": "X_pow",
                        "Y": "Y_pow",
                        "Z": "Z_pow",
                        "H": "H_pow",
                        "CZ": "CZ_pow",
                        "CX": "CX_pow",
                        "iSwap": "iSWAP_pow",
                    }.get(base_key_for_pow, f"{base_key_for_pow}_pow")
            else:
                style_key = base_key_for_pow

        style_opts_str = self.gate_styles.get(style_key, "")
        if not style_opts_str:
            if gate_type.__name__ == "FSimGate":
                style_opts_str = self.gate_styles.get("FSim", "")
            elif gate_type.__name__ == "PhasedXZGate":
                style_opts_str = self.gate_styles.get("PhasedXZ", "")

        final_style_tikz = f"[{style_opts_str}]" if style_opts_str else ""

        # Apply special Quantikz commands for specific gate types
        if isinstance(gate, ops.MeasurementGate):
            lbl = gate.key.replace("_", r"\_") if gate.key else ""
            for i in q_indices:
                output[i] = f"\\meter{final_style_tikz}{{{lbl}}}"
            return output
        if isinstance(gate, ops.CNotPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            c, t = (
                (self.qubit_to_index[op.qubits[0]], self.qubit_to_index[op.qubits[1]])
                if len(op.qubits) == 2
                else (q_indices[0], q_indices[0])
            )
            output[c], output[t] = (
                f"\\ctrl{final_style_tikz}{{{t-c}}}",
                f"\\targ{final_style_tikz}{{}}",
            )
            return output
        if isinstance(gate, ops.CZPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            i1, i2 = (
                (q_indices[0], q_indices[1])
                if len(q_indices) >= 2
                else (q_indices[0], q_indices[0])
            )
            output[i1], output[i2] = (
                f"\\ctrl{final_style_tikz}{{{i2-i1}}}",
                f"\\control{final_style_tikz}{{}}",
            )
            return output
        if isinstance(gate, ops.SwapPowGate) and hasattr(gate, "exponent") and gate.exponent == 1:
            i1, i2 = (
                (q_indices[0], q_indices[1])
                if len(q_indices) >= 2
                else (q_indices[0], q_indices[0])
            )
            output[i1], output[i2] = (
                f"\\swap{final_style_tikz}{{{i2-i1}}}",
                f"\\targX{final_style_tikz}{{}}",
            )
            return output

        # Handle generic \gate command for single and multi-qubit gates
        if not q_indices:
            warnings.warn(f"Op {op} has no qubits.")
            return output
        if len(q_indices) == 1:
            output[q_indices[0]] = f"\\gate{final_style_tikz}{{{gate_name_render}}}"
        else:  # Multi-qubit gate
            wires_opt = f"wires={q_indices[-1]-q_indices[0]+1}"
            if style_opts_str:
                combined_opts = f"{wires_opt}, {style_opts_str}"
            else:
                combined_opts = wires_opt
            output[q_indices[0]] = f"\\gate[{combined_opts}]{{{gate_name_render}}}"
            for i in range(1, len(q_indices)):
                output[q_indices[i]] = "\\qw"
        return output

    def _generate_latex_body(self) -> str:
        """Generates the main LaTeX body for the circuit diagram.

        Iterates through the circuit's moments, renders each operation, and
        arranges them into Quantikz environments. Supports circuit folding
        into multiple rows if `fold_at` is specified.
        Handles qubit wire labels and ensures correct LaTeX syntax.
        """
        chunks, m_count, active_chunk = [], 0, [[] for _ in range(self.num_qubits)]
        # Add initial wire labels for the first chunk
        for i in range(self.num_qubits):
            active_chunk[i].append(f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i], i)}}}")

        for m_idx, moment in enumerate(self.circuit):
            m_count += 1
            moment_out = ["\\qw"] * self.num_qubits
            processed_indices = set()

            for op in moment:
                q_idx_op = sorted([self.qubit_to_index[q] for q in op.qubits])
                if not q_idx_op:
                    warnings.warn(f"Op {op} no qubits.")
                    continue
                if any(q in processed_indices for q in q_idx_op):
                    for q_idx in q_idx_op:
                        if q_idx not in processed_indices:
                            moment_out[q_idx] = "\\qw"
                    continue
                op_rnd = self._render_operation(op)
                for idx, tex in op_rnd.items():
                    if idx not in processed_indices:
                        moment_out[idx] = tex
                processed_indices.update(q_idx_op)
            for i in range(self.num_qubits):
                active_chunk[i].append(moment_out[i])

            is_last_m = m_idx == len(self.circuit) - 1
            if self.fold_at and m_count % self.fold_at == 0 and not is_last_m:
                for i in range(self.num_qubits):
                    lbl = self._get_wire_label(self.sorted_qubits[i], i)
                    active_chunk[i].extend([f"\\rstick{{{lbl}}}", "\\qw"])
                chunks.append(active_chunk)
                active_chunk = [[] for _ in range(self.num_qubits)]
                for i in range(self.num_qubits):
                    active_chunk[i].append(
                        f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}"
                    )

        if self.num_qubits > 0:
            ended_on_fold = self.fold_at and m_count > 0 and m_count % self.fold_at == 0
            if not ended_on_fold or not self.fold_at:
                for i in range(self.num_qubits):
                    if not active_chunk[i]:
                        active_chunk[i] = [
                            f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}"
                        ]
                    active_chunk[i].append("\\qw")
            if self.fold_at:
                for i in range(self.num_qubits):
                    if not active_chunk[i]:
                        active_chunk[i] = [
                            f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}"
                        ]
                    active_chunk[i].extend(
                        [f"\\rstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}}", "\\qw"]
                    )
        chunks.append(active_chunk)

        final_parts = []
        opts_str = self._get_quantikz_options_string()
        for chunk_data in chunks:
            if not any(row for row_list in chunk_data for row in row_list):
                continue

            is_empty_like = True
            if chunk_data and any(chunk_data):
                for r_cmds in chunk_data:
                    if any(
                        cmd not in ["\\qw", ""]
                        and not cmd.startswith("\\lstick")
                        and not cmd.startswith("\\rstick")
                        for cmd in r_cmds
                    ):
                        is_empty_like = False
                        break
                if all(
                    all(
                        cmd == "\\qw" or cmd.startswith("\\lstick") or cmd.startswith("\\rstick")
                        for cmd in r
                    )
                    for r in chunk_data
                    if r
                ):
                    if len(chunks) > 1 or not self.circuit:
                        if all(len(r) <= (4 if self.fold_at else 2) for r in chunk_data if r):
                            is_empty_like = True
            if is_empty_like and len(chunks) > 1 and self.circuit:
                continue

            lines = [f"\\begin{{quantikz}}{opts_str}"]
            for i in range(self.num_qubits):
                if i < len(chunk_data) and chunk_data[i]:
                    lines.append(" & ".join(chunk_data[i]) + " \\\\")
                elif i < self.num_qubits:
                    lines.append(
                        f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}} & \\qw \\\\"
                    )

            if len(lines) > 1:
                for k_idx in range(len(lines) - 1, 0, -1):
                    if lines[k_idx].strip() and lines[k_idx].strip() != "\\\\":
                        if lines[k_idx].endswith(" \\\\"):
                            lines[k_idx] = lines[k_idx].rstrip()[:-3].rstrip()
                        break
                    elif lines[k_idx].strip() == "\\\\" and k_idx == len(lines) - 1:
                        lines[k_idx] = ""
            lines.append("\\end{quantikz}")
            final_parts.append("\n".join(filter(None, lines)))

        if not final_parts and self.num_qubits > 0:
            lines = [f"\\begin{{quantikz}}{opts_str}"]
            for i in range(self.num_qubits):
                lines.append(
                    f"\\lstick{{{self._get_wire_label(self.sorted_qubits[i],i)}}} & \\qw"
                    + (" \\\\" if i < self.num_qubits - 1 else "")
                )
            lines.append("\\end{quantikz}")
            return "\n".join(lines)
        return "\n\n\\vspace{1em}\n\n".join(final_parts)

    def generate_latex_document(self, preamble_template: Optional[str] = None) -> str:
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
        preamble += f"\n% --- Custom Preamble Injection Point ---\n{self.custom_preamble}\n% --- End Custom Preamble ---\n"
        doc_parts = [preamble, "\\begin{document}", self._generate_latex_body()]
        if self.custom_postamble:
            doc_parts.extend(
                [
                    "\n% --- Custom Postamble Start ---",
                    self.custom_postamble,
                    "% --- Custom Postamble End ---\n",
                ]
            )
        doc_parts.append("\\end{document}")
        return "\n".join(doc_parts)
