# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, List, Tuple, Iterator, Dict, Optional

import sympy
import urllib.parse
import json

import cirq
from cirq import ops
from cirq.contrib.quirk.quirk_parse_gates import (ParityControlCell, Cell,
                                                  QuirkPseudoSwapOperation,
                                                  SetDefaultInputCell)
from cirq.contrib.quirk.quirk_gate_reg_utils import (
    popcnt, mod_inv_else_1, reg_control, reg_input_family,
    reg_unsupported_gates, reg_gate, reg_const, reg_formula_gate,
    reg_parameterized_gate, reg_ignored_family, reg_ignored_gate,
    reg_arithmetic_gate, reg_arithmetic_family,
    reg_size_dependent_arithmetic_family, reg_unsupported_family, reg_family,
    reg_bit_permutation_family, deinterleave_bit, interleave_bit, CellType,
    reg_modular_arithmetic_family, invertible_else_1, CellArgs, reg_measurement)


def quirk_url_to_circuit(quirk_url: str) -> 'cirq.Circuit':

    parsed_url = urllib.parse.urlparse(quirk_url)
    if not parsed_url.fragment:
        return cirq.Circuit()

    if not parsed_url.fragment.startswith('circuit='):
        raise ValueError('Not a valid quirk url. The URL fragment (the part '
                         'after the #) must start with "circuit=".\n'
                         f'URL={quirk_url}')

    # URL parser may not have fixed escaped characters in the fragment.
    json_text = parsed_url.fragment[len('circuit='):]
    if '%22' in json_text:
        json_text = urllib.parse.unquote(json_text)

    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError('Circuit JSON must have a top-level dictionary.\n'
                         f'URL={quirk_url}')
    if not data.keys() <= {'cols', 'gates', 'init'}:
        raise ValueError(f'Unrecognized Circuit JSON keys.\nURL={quirk_url}')
    if 'gates' in data:
        raise NotImplementedError('Custom gates not supported yet.\n'
                                  f'URL={quirk_url}')
    if 'init' in data:
        raise NotImplementedError('Custom initial states not supported yet.\n'
                                  f'URL={quirk_url}')
    if 'cols' not in data:
        raise ValueError('Circuit JSON dictionary must have a "cols" entry.\n'
                         f'URL={quirk_url}')

    cols = data['cols']
    if not isinstance(cols, list):
        raise ValueError('Circuit JSON cols must be a list.\n'
                         f'URL={quirk_url}')

    # Parse column json into cells.
    registry = {entry.identifier: entry for entry in _gate_registry()}
    parsed_cols: List[List[Cell]] = []
    for i, col in enumerate(cols):
        parsed_cols.append(parse_col_cells(registry, i, col))

    # Apply column modifiers (controls and inputs).
    for c in parsed_cols:
        for i in range(len(c)):
            if c[i] is not None:
                c[i].modify_column(c)

    # Apply persistent modifiers (classical assignments).
    persistent_mods = {}
    for c in parsed_cols:
        for cell in c:
            if cell is not None:
                for key, modifier in cell.persistent_modifiers().items():
                    persistent_mods[key] = modifier
        for i in range(len(c)):
            for modifier in persistent_mods.values():
                c[i] = modifier(c[i])

    # Extract circuit operations from modified cells.
    result = cirq.Circuit()
    for col in parsed_cols:
        basis_change = cirq.Circuit.from_ops(
            cell.basis_change() for cell in col if cell is not None)
        body = (cell.operations() for cell in col if cell is not None)
        result += basis_change
        result += body
        result += basis_change**-1

    return result


def parse_col_cells(registry: Dict[str, CellType], col: int,
                    col_data: Any) -> List[Optional[Cell]]:
    if not isinstance(col_data, list):
        raise ValueError('col must be a list.\ncol: {!r}'.format(col_data))
    return [
        parse_cell(registry, row, col, col_data[row])
        for row in range(len(col_data))
    ]


def parse_cell(registry: Dict[str, CellType], row: int, col: int,
               entry: Any) -> Optional[Cell]:
    if entry == 1:
        return None

    key = None
    arg = None
    if isinstance(entry, dict):
        key = entry['id']
        arg = entry.get('arg', None)
    elif isinstance(entry, str):
        key = entry

    if isinstance(key, str) and key in registry:
        _, size, func = registry[key]
        qubits = cirq.LineQubit.range(row, row + size)
        return func(CellArgs(qubits, arg, row=row, col=col))

    raise ValueError('Unrecognized column entry: {!r}'.format(entry))


def _gate_registry() -> Iterator[CellType]:
    # Swap.
    yield CellType("Swap",
                   1, lambda args: QuirkPseudoSwapOperation(args.qubits, []))

    # Controls.
    yield from reg_control("•", None)
    yield from reg_control("◦", ops.X)
    yield from reg_control("⊕", ops.Y**0.5)
    yield from reg_control("⊖", ops.Y**-0.5)
    yield from reg_control("⊗", ops.X**-0.5)
    yield from reg_control("(/)", ops.X**0.5)

    # Parity controls.
    yield CellType(
        "xpar", 1, lambda args: ParityControlCell(args.qubits, (ops.Y**0.5).
                                                  on_each(args.qubits)))
    yield CellType(
        "ypar", 1, lambda args: ParityControlCell(args.qubits, (cirq.X**-0.5).
                                                  on_each(args.qubits)))
    yield CellType("zpar", 1, lambda args: ParityControlCell(args.qubits, []))

    # Input gates.
    yield from reg_input_family("inputA", "a")
    yield from reg_input_family("inputB", "b")
    yield from reg_input_family("inputR", "r")
    yield from reg_input_family("revinputA", "a", rev=True)
    yield from reg_input_family("revinputB", "b", rev=True)
    yield CellType("setA", 2, lambda args: SetDefaultInputCell('a', args.value))
    yield CellType("setB", 2, lambda args: SetDefaultInputCell('b', args.value))
    yield CellType("setR", 2, lambda args: SetDefaultInputCell('r', args.value))

    # Post selection.
    yield from reg_unsupported_gates(
        "|0⟩⟨0|",
        "|1⟩⟨1|",
        "|+⟩⟨+|",
        "|-⟩⟨-|",
        "|X⟩⟨X|",
        "|/⟩⟨/|",
        "0",
        reason='postselection is not implemented in Cirq')

    # Non-physical operations.
    yield from reg_unsupported_gates("__error__",
                                     "__unstable__UniversalNot",
                                     reason="Unphysical operation.")

    # Scalars.
    yield from reg_gate("…", gate=ops.IdentityGate(num_qubits=1))
    yield from reg_const("NeGate", ops.GlobalPhaseOperation(-1))
    yield from reg_const("i", ops.GlobalPhaseOperation(1j))
    yield from reg_const("-i", ops.GlobalPhaseOperation(-1j))
    yield from reg_const("√i", ops.GlobalPhaseOperation(1j**0.5))
    yield from reg_const("√-i", ops.GlobalPhaseOperation((-1j)**0.5))

    # Measurement.
    yield from reg_measurement("Measure")
    yield from reg_measurement("ZDetector")
    yield from reg_measurement("YDetector", basis_change=ops.X**-0.5)
    yield from reg_measurement("XDetector", basis_change=ops.Y**0.5)
    yield from reg_unsupported_gates(
        "XDetectControlReset",
        "YDetectControlReset",
        "ZDetectControlReset",
        reason="Classical feedback is not implemented in Cirq")

    # Fixed single qubit rotations.
    yield from reg_gate("H", gate=ops.H)
    yield from reg_gate("X", gate=ops.X)
    yield from reg_gate("Y", gate=ops.Y)
    yield from reg_gate("Z", gate=ops.Z)
    yield from reg_gate("X^½", gate=ops.X**(1 / 2))
    yield from reg_gate("X^⅓", gate=ops.X**(1 / 3))
    yield from reg_gate("X^¼", gate=ops.X**(1 / 4))
    yield from reg_gate("X^⅛", gate=ops.X**(1 / 8))
    yield from reg_gate("X^⅟₁₆", gate=ops.X**(1 / 16))
    yield from reg_gate("X^⅟₃₂", gate=ops.X**(1 / 32))
    yield from reg_gate("X^-½", gate=ops.X**(-1 / 2))
    yield from reg_gate("X^-⅓", gate=ops.X**(-1 / 3))
    yield from reg_gate("X^-¼", gate=ops.X**(-1 / 4))
    yield from reg_gate("X^-⅛", gate=ops.X**(-1 / 8))
    yield from reg_gate("X^-⅟₁₆", gate=ops.X**(-1 / 16))
    yield from reg_gate("X^-⅟₃₂", gate=ops.X**(-1 / 32))
    yield from reg_gate("Y^½", gate=ops.Y**(1 / 2))
    yield from reg_gate("Y^⅓", gate=ops.Y**(1 / 3))
    yield from reg_gate("Y^¼", gate=ops.Y**(1 / 4))
    yield from reg_gate("Y^⅛", gate=ops.Y**(1 / 8))
    yield from reg_gate("Y^⅟₁₆", gate=ops.Y**(1 / 16))
    yield from reg_gate("Y^⅟₃₂", gate=ops.Y**(1 / 32))
    yield from reg_gate("Y^-½", gate=ops.Y**(-1 / 2))
    yield from reg_gate("Y^-⅓", gate=ops.Y**(-1 / 3))
    yield from reg_gate("Y^-¼", gate=ops.Y**(-1 / 4))
    yield from reg_gate("Y^-⅛", gate=ops.Y**(-1 / 8))
    yield from reg_gate("Y^-⅟₁₆", gate=ops.Y**(-1 / 16))
    yield from reg_gate("Y^-⅟₃₂", gate=ops.Y**(-1 / 32))
    yield from reg_gate("Z^½", gate=ops.Z**(1 / 2))
    yield from reg_gate("Z^⅓", gate=ops.Z**(1 / 3))
    yield from reg_gate("Z^¼", gate=ops.Z**(1 / 4))
    yield from reg_gate("Z^⅛", gate=ops.Z**(1 / 8))
    yield from reg_gate("Z^⅟₁₆", gate=ops.Z**(1 / 16))
    yield from reg_gate("Z^⅟₃₂", gate=ops.Z**(1 / 32))
    yield from reg_gate("Z^⅟₆₄", gate=ops.Z**(1 / 64))
    yield from reg_gate("Z^⅟₁₂₈", gate=ops.Z**(1 / 128))
    yield from reg_gate("Z^-½", gate=ops.Z**(-1 / 2))
    yield from reg_gate("Z^-⅓", gate=ops.Z**(-1 / 3))
    yield from reg_gate("Z^-¼", gate=ops.Z**(-1 / 4))
    yield from reg_gate("Z^-⅛", gate=ops.Z**(-1 / 8))
    yield from reg_gate("Z^-⅟₁₆", gate=ops.Z**(-1 / 16))

    # Dynamic single qubit rotations.
    yield from reg_gate("X^t", gate=ops.X**sympy.Symbol('t'))
    yield from reg_gate("Y^t", gate=ops.Y**sympy.Symbol('t'))
    yield from reg_gate("Z^t", gate=ops.Z**sympy.Symbol('t'))
    yield from reg_gate("X^-t", gate=ops.X**-sympy.Symbol('t'))
    yield from reg_gate("Y^-t", gate=ops.Y**-sympy.Symbol('t'))
    yield from reg_gate("Z^-t", gate=ops.Z**-sympy.Symbol('t'))
    yield from reg_gate("e^iXt", gate=ops.Rx(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^iYt", gate=ops.Ry(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^iZt", gate=ops.Rz(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iXt",
                        gate=ops.Rx(-2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iYt",
                        gate=ops.Ry(-2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iZt",
                        gate=ops.Rz(-2 * sympy.pi * sympy.Symbol('t')))

    # Classically parameterized single qubit rotations.
    yield from reg_formula_gate("X^ft", "sin(pi*t)", lambda e: ops.X**e)
    yield from reg_formula_gate("Y^ft", "sin(pi*t)", lambda e: ops.Y**e)
    yield from reg_formula_gate("Z^ft", "sin(pi*t)", lambda e: ops.Z**e)
    yield from reg_formula_gate("Rxft", "pi*t*t", lambda e: ops.Rx(e))
    yield from reg_formula_gate("Ryft", "pi*t*t", lambda e: ops.Ry(e))
    yield from reg_formula_gate("Rzft", "pi*t*t", lambda e: ops.Rz(e))

    # Quantum parameterized single qubit rotations.
    yield from reg_parameterized_gate("X^(A/2^n)", ops.X, +1)
    yield from reg_parameterized_gate("Y^(A/2^n)", ops.Y, +1)
    yield from reg_parameterized_gate("Z^(A/2^n)", ops.Z, +1)
    yield from reg_parameterized_gate("X^(-A/2^n)", ops.X, -1)
    yield from reg_parameterized_gate("Y^(-A/2^n)", ops.Y, -1)
    yield from reg_parameterized_gate("Z^(-A/2^n)", ops.Z, -1)

    # Displays.
    yield from reg_ignored_family("Amps")
    yield from reg_ignored_family("Chance")
    yield from reg_ignored_family("Sample")
    yield from reg_ignored_family("Density")
    yield from reg_ignored_gate("Bloch")

    # Arithmetic.
    yield from reg_arithmetic_gate("^A<B", 1, lambda x, a, b: x ^ int(a < b))
    yield from reg_arithmetic_gate("^A>B", 1, lambda x, a, b: x ^ int(a > b))
    yield from reg_arithmetic_gate("^A<=B", 1, lambda x, a, b: x ^ int(a <= b))
    yield from reg_arithmetic_gate("^A>=B", 1, lambda x, a, b: x ^ int(a >= b))
    yield from reg_arithmetic_gate("^A=B", 1, lambda x, a, b: x ^ int(a == b))
    yield from reg_arithmetic_gate("^A!=B", 1, lambda x, a, b: x ^ int(a != b))
    yield from reg_arithmetic_family("inc", lambda x: x + 1)
    yield from reg_arithmetic_family("dec", lambda x: x - 1)
    yield from reg_arithmetic_family("+=A", lambda x, a: x + a)
    yield from reg_arithmetic_family("-=A", lambda x, a: x - a)
    yield from reg_arithmetic_family("+=AA", lambda x, a: x + a * a)
    yield from reg_arithmetic_family("-=AA", lambda x, a: x - a * a)
    yield from reg_arithmetic_family("+=AB", lambda x, a, b: x + a * b)
    yield from reg_arithmetic_family("-=AB", lambda x, a, b: x - a * b)
    yield from reg_arithmetic_family("^=A", lambda x, a: x ^ a)
    yield from reg_arithmetic_family("+cntA", lambda x, a: x + popcnt(a))
    yield from reg_arithmetic_family("-cntA", lambda x, a: x - popcnt(a))
    yield from reg_arithmetic_family(
        "Flip<A", lambda x, a: a - x - 1 if x < a else x)
    yield from reg_arithmetic_family("*A", lambda x, a: x * a if a & 1 else x)
    yield from reg_size_dependent_arithmetic_family(
        "/A", lambda n: lambda x, a: x * mod_inv_else_1(a, 1 << n))

    # Modular arithmetic.
    yield from reg_modular_arithmetic_family("incmodR", lambda x, r: x + 1)
    yield from reg_modular_arithmetic_family("decmodR", lambda x, r: x - 1)
    yield from reg_modular_arithmetic_family("+AmodR", lambda x, a, r: x + a)
    yield from reg_modular_arithmetic_family("-AmodR", lambda x, a, r: x - a)
    yield from reg_modular_arithmetic_family(
        "+ABmodR", lambda x, a, b, r: x + a * b)
    yield from reg_modular_arithmetic_family(
        "-ABmodR", lambda x, a, b, r: x - a * b)
    yield from reg_modular_arithmetic_family(
        "*AmodR", lambda x, a, r: x * invertible_else_1(a, r))
    yield from reg_modular_arithmetic_family(
        "/AmodR", lambda x, a, r: x * mod_inv_else_1(a, r))
    yield from reg_modular_arithmetic_family(
        "*BToAmodR", lambda x, a, b, r: x * pow(invertible_else_1(b, r), a, r))
    yield from reg_modular_arithmetic_family(
        "/BToAmodR", lambda x, a, b, r: x * pow(mod_inv_else_1(b, r), a, r))

    # Dynamic gates with discretized actions.
    yield from reg_unsupported_gates("X^⌈t⌉",
                                     "X^⌈t-¼⌉",
                                     reason="discrete parameter")
    yield from reg_unsupported_family("Counting", reason="discrete parameter")
    yield from reg_unsupported_family("Uncounting", reason="discrete parameter")
    yield from reg_unsupported_family(">>t", reason="discrete parameter")
    yield from reg_unsupported_family("<<t", reason="discrete parameter")

    # Gates that are no longer in the toolbox and have dominant replacements.
    yield from reg_unsupported_family("add",
                                      reason="deprecated; use +=A instead")
    yield from reg_unsupported_family("sub",
                                      reason="deprecated; use -=A instead")
    yield from reg_unsupported_family("c+=ab",
                                      reason="deprecated; use +=AB instead")
    yield from reg_unsupported_family("c-=ab",
                                      reason="deprecated; use -=AB instead")

    # Frequency space.
    yield from reg_family("QFT", lambda n: cirq.QuantumFourierTransformGate(n))
    yield from reg_family(
        "QFT†", lambda n: cirq.inverse(cirq.QuantumFourierTransformGate(n)))
    yield from reg_family(
        "PhaseGradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
                                                          exponent=0.5))
    yield from reg_family(
        "PhaseUngradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
                                                            exponent=-0.5))
    yield from reg_family(
        "grad^t", lambda n: cirq.PhaseGradientGate(
            num_qubits=n, exponent=2**(n - 1) * sympy.Symbol('t')))
    yield from reg_family(
        "grad^-t", lambda n: cirq.PhaseGradientGate(
            num_qubits=n, exponent=-2**(n - 1) * sympy.Symbol('t')))

    # Bit level permutations.
    yield from reg_bit_permutation_family(
        "<<", 'left_rotate', lambda n, x: (x + 1) % n)
    yield from reg_bit_permutation_family(
        ">>", 'right_rotate', lambda n, x: (x - 1) % n)
    yield from reg_bit_permutation_family("rev",
                                          'reverse', lambda n, x: n - x - 1)
    yield from reg_bit_permutation_family("weave", 'interleave', interleave_bit)
    yield from reg_bit_permutation_family("split", 'deinterleave',
                                          deinterleave_bit)
