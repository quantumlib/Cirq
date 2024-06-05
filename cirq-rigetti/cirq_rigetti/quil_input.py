# Copyright 2020 The Cirq Developers
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

from typing import Any, Callable, Type, cast, Dict, Union, List, Tuple, Optional

import sympy
import numpy as np
from numpy.typing import NDArray

from pyquil.quil import Program
from pyquil.quilbase import (
    Declare,
    DefGate,
    Gate as PyQuilGate,
    Measurement as PyQuilMeasurement,
    Pragma,
    Reset,
    ResetQubit,
    Fence,
    FenceAll,
)
from pyquil.quilatom import (
    MemoryReference,
    ParameterDesignator,
    QubitDesignator,
    Function,
    BinaryExp,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Parameter,
    substitute_array,
    qubit_index,
)
from pyquil.simulation import matrices

import cirq
from cirq.circuits.circuit import Circuit
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs, CircuitDiagramInfo
from cirq.devices.line_qubit import LineQubit
from cirq.devices.noise_utils import OpIdentifier
from cirq.value import value_equality
from cirq.protocols import is_parameterized

from cirq.ops.common_gates import CNOT, CZ, CZPowGate, H, S, T, ZPowGate, YPowGate, XPowGate
from cirq.ops.parity_gates import ZZPowGate, XXPowGate, YYPowGate
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.fsim_gate import FSimGate, PhasedFSimGate
from cirq.ops.identity import I
from cirq.ops.matrix_gates import MatrixGate
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.swap_gates import ISWAP, ISwapPowGate, SWAP
from cirq.ops.three_qubit_gates import CCNOT, CSWAP
from cirq.ops.raw_types import Gate
from cirq.ops.kraus_channel import KrausChannel
from cirq._compat import cached_method


class UndefinedQuilGate(Exception):
    """Error for a undefined Quil Gate."""


class UnsupportedQuilInstruction(Exception):
    """Error for a unsupported instruction."""


#
# Cirq doesn't have direct analogues of these Quil gates
#


@value_equality(distinct_child_types=True, approximate=True)
class CPHASE00(Gate):
    """Cirq equivalent to Quil CPHASE00."""

    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return matrices.CPHASE00(self.phi)

    def _circuit_diagram_info_(
        self, args: CircuitDiagramInfoArgs
    ) -> CircuitDiagramInfo:  # pragma: no cover
        return CircuitDiagramInfo(wire_symbols=("@00", "@00"), exponent=self.phi / np.pi)

    def __repr__(self) -> str:  # pragma: no cover
        """Represent the CPHASE gate as a string."""
        return f"CPHASE00({self.phi:.3f})"

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> Gate:  # pragma: no cover
        return type(self)(phi=resolver.value_of(self.phi, recursive))

    def _is_parameterized_(self) -> bool:  # pragma: no cover
        parameter_names = ["phi"]
        return any(is_parameterized(getattr(self, p)) for p in parameter_names)

    def _value_equality_values_(self):  # pragma: no cover
        return (self.phi,)

    def _value_equality_approximate_values_(self):  # pragma: no cover
        return (self.phi,)


@value_equality(distinct_child_types=True, approximate=True)
class CPHASE01(Gate):
    """Cirq equivalent to Quil CPHASE01."""

    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return matrices.CPHASE01(self.phi)

    def _circuit_diagram_info_(
        self, args: CircuitDiagramInfoArgs
    ) -> CircuitDiagramInfo:  # pragma: no cover
        return CircuitDiagramInfo(wire_symbols=("@01", "@01"), exponent=self.phi / np.pi)

    def __repr__(self) -> str:  # pragma: no cover
        """Represent the CPHASE gate as a string."""
        return f"CPHASE01({self.phi:.3f})"

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> Gate:  # pragma: no cover
        return type(self)(phi=resolver.value_of(self.phi, recursive))

    def _is_parameterized_(self) -> bool:  # pragma: no cover
        parameter_names = ["phi"]
        return any(is_parameterized(getattr(self, p)) for p in parameter_names)

    def _value_equality_values_(self):  # pragma: no cover
        return (self.phi,)

    def _value_equality_approximate_values_(self):  # pragma: no cover
        return (self.phi,)


@value_equality(distinct_child_types=True, approximate=True)
class CPHASE10(Gate):
    """Cirq equivalent to Quil CPHASE10."""

    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return matrices.CPHASE10(self.phi)

    def _circuit_diagram_info_(
        self, args: CircuitDiagramInfoArgs
    ) -> CircuitDiagramInfo:  # pragma: no cover
        return CircuitDiagramInfo(wire_symbols=("@10", "@10"), exponent=self.phi / np.pi)

    def __repr__(self) -> str:  # pragma: no cover
        """Represent the CPHASE gate as a string."""
        return f"CPHASE10({self.phi:.3f})"

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> Gate:  # pragma: no cover
        return type(self)(phi=resolver.value_of(self.phi, recursive))

    def _is_parameterized_(self) -> bool:  # pragma: no cover
        parameter_names = ["phi"]
        return any(is_parameterized(getattr(self, p)) for p in parameter_names)

    def _value_equality_values_(self):  # pragma: no cover
        return (self.phi,)

    def _value_equality_approximate_values_(self):  # pragma: no cover
        return (self.phi,)


@value_equality(distinct_child_types=True, approximate=True)
class PSWAP(Gate):
    """Cirq equivalent to Quil PSWAP."""

    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return matrices.PSWAP(self.phi)

    def _circuit_diagram_info_(
        self, args: CircuitDiagramInfoArgs
    ) -> CircuitDiagramInfo:  # pragma: no cover
        return CircuitDiagramInfo(wire_symbols=("PSWAP", "PSWAP"), exponent=self.phi / np.pi)

    def __repr__(self) -> str:  # pragma: no cover
        """Represent the PSWAP gate as a string."""
        return f"PSWAP({self.phi:.3f})"

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> Gate:  # pragma: no cover
        return type(self)(phi=resolver.value_of(self.phi, recursive))

    def _is_parameterized_(self) -> bool:  # pragma: no cover
        parameter_names = ["phi"]
        return any(is_parameterized(getattr(self, p)) for p in parameter_names)

    def _value_equality_values_(self):  # pragma: no cover
        return (self.phi,)

    def _value_equality_approximate_values_(self):  # pragma: no cover
        return (self.phi,)


PRAGMA_ERROR = """
Please remove PRAGMAs from your Quil program.
If you would like to add noise, do so after conversion.
"""

RESET_ERROR = """
Please remove RESETs from your Quil program.
RESET directives have special meaning on QCS, to enable active reset.
"""


# Parameterized gates map to functions that produce Gate constructors.
SUPPORTED_GATES: Dict[str, Union[Gate, Callable[..., Gate]]] = {
    "CCNOT": CCNOT,
    "CNOT": CNOT,
    "CSWAP": CSWAP,
    "CPHASE": CZPowGate,
    "CPHASE00": CPHASE00,
    "CPHASE01": CPHASE01,
    "CPHASE10": CPHASE10,
    "PSWAP": PSWAP,
    "CZ": CZ,
    "PHASE": ZPowGate,
    "H": H,
    "I": I,
    "ISWAP": ISWAP,
    "RX": XPowGate,
    "RY": YPowGate,
    "RZ": ZPowGate,
    "S": S,
    "SWAP": SWAP,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
    "XY": ISwapPowGate,
    "RZZ": ZZPowGate,
    "RYY": YYPowGate,
    "RXX": XXPowGate,
    "FSIM": FSimGate,
    "PHASEDFSIM": PhasedFSimGate,
}

# Gate parameters must be transformed to Cirq units
PARAMETRIC_TRANSFORMERS: Dict[str, Callable] = {
    "CPHASE": lambda theta: dict(exponent=theta / np.pi, global_shift=0.0),
    "CPHASE00": lambda phi: dict(phi=phi),
    "CPHASE01": lambda phi: dict(phi=phi),
    "CPHASE10": lambda phi: dict(phi=phi),
    "PSWAP": lambda phi: dict(phi=phi),
    "PHASE": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.0),
    "XY": lambda theta: dict(exponent=theta / np.pi, global_shift=0.0),
    "RX": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "RY": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "RZ": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "RXX": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "RYY": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "RZZ": lambda theta: dict(exponent=theta / np.pi, global_shift=-0.5),
    "FSIM": lambda theta, phi: dict(theta=-1 * theta / 2, phi=-1 * phi),
    "PHASEDFSIM": lambda theta, zeta, chi, gamma, phi: dict(
        theta=-1 * theta / 2, zeta=zeta, chi=chi, gamma=gamma, phi=-1 * phi
    ),
}


def circuit_from_quil(quil: Union[str, Program]) -> Circuit:
    """Convert a Quil program to a Cirq Circuit.

    Args:
        quil: The Quil program to convert.

    Returns:
        A Cirq Circuit generated from the Quil program.

    Raises:
        UnsupportedQuilInstruction: Cirq does not support the specified Quil instruction.
        UndefinedQuilGate: Cirq does not support the specified Quil gate.

    References:
        https://github.com/rigetti/pyquil
    """
    if isinstance(quil, str):
        program = Program(quil)
    else:
        program = quil
    circuit = Circuit()

    defined_gates, parameter_transformers = get_defined_gates(program)

    kraus_model: Dict[Tuple[QubitDesignator, ...], List[NDArray[np.complex_]]] = {}
    confusion_maps: Dict[int, NDArray[np.float_]] = {}

    # Interpret the Pragmas
    for inst in program:
        if not isinstance(inst, Pragma):  # pragma: no cover
            continue

        # ADD-KRAUS provides Kraus operators that replace the gate operation
        if inst.command == "ADD-KRAUS":  # pragma: no cover
            args = inst.args
            gate_name = str(args[0])
            if gate_name in matrices.QUANTUM_GATES:
                u = matrices.QUANTUM_GATES[gate_name]
            elif gate_name in defined_gates:
                u = defined_gates[gate_name]
            else:
                raise UndefinedQuilGate(f"{gate_name} is not known.")

            entries = np.fromstring(
                inst.freeform_string.strip("()").replace("i", "j"), dtype=np.complex_, sep=" "
            )
            dim = int(np.sqrt(len(entries)))
            kraus_gate_op = entries.reshape((dim, dim))

            kraus_op = remove_gate_from_kraus([kraus_gate_op], u)[0]

            if args in kraus_model:
                kraus_model[args].append(kraus_op)
            else:
                kraus_model[args] = [kraus_op]

        # READOUT-POVM provides a confusion matrix
        elif inst.command == "READOUT-POVM":
            qubit = qubit_index(inst.args[0])
            entries = np.fromstring(
                inst.freeform_string.strip("()").replace("i", "j"), dtype=np.float_, sep=" "
            )
            confusion_matrix = entries.reshape((2, 2)).T

            # these types actually agree - both arrays are floats
            confusion_maps[qubit] = confusion_matrix  # type: ignore

        else:
            raise UnsupportedQuilInstruction(PRAGMA_ERROR)  # pragma: no cover

    # Interpret the instructions
    for inst in program:
        # Pass when encountering a DECLARE.
        if isinstance(inst, Declare):
            pass

        # Convert pyQuil gates to Cirq operations.
        elif isinstance(inst, PyQuilGate):
            quil_gate_name = inst.name
            quil_gate_params = inst.params
            line_qubits = list(LineQubit(qubit_index(q)) for q in inst.qubits)
            if quil_gate_name not in defined_gates:
                raise UndefinedQuilGate(f"Quil gate {quil_gate_name} not supported in Cirq.")
            cirq_gate_fn = defined_gates[quil_gate_name]
            if quil_gate_params:
                params = [quil_expression_to_sympy(p) for p in quil_gate_params]
                transformer = parameter_transformers[quil_gate_name]
                circuit += cast(Callable[..., Gate], cirq_gate_fn)(**transformer(*params))(
                    *line_qubits
                )
            else:
                circuit += cirq_gate_fn(*line_qubits)

        # Convert pyQuil MEASURE operations to Cirq MeasurementGate objects.
        elif isinstance(inst, PyQuilMeasurement):
            qubit = qubit_index(inst.qubit)
            line_qubit = LineQubit(qubit)
            if inst.classical_reg is None:
                raise UnsupportedQuilInstruction(
                    f"Quil measurement {inst} without classical register "
                    f"not currently supported in Cirq."
                )
            quil_memory_reference = inst.classical_reg.out()
            if qubit in confusion_maps:
                cmap: Dict[Tuple[int, ...], NDArray[np.float_]] = {(qubit,): confusion_maps[qubit]}
                """
                Argument "confusion_map" to "MeasurementGate" has incompatible type
                    "         Dict[Tuple[int],      ndarray[Any, dtype[floating[Any]]]]"
                expected
                    "Optional[Dict[Tuple[int, ...], ndarray[Any, Any]]]"
                """
                circuit += MeasurementGate(1, key=quil_memory_reference, confusion_map=cmap)(
                    line_qubit
                )
            else:
                circuit += MeasurementGate(1, key=quil_memory_reference)(line_qubit)

        # PRAGMAs
        elif isinstance(inst, Pragma):
            continue

        # Drop FENCE statements
        elif isinstance(inst, (Fence, FenceAll)):  # pragma: no cover
            continue

        # Drop DEFGATES
        elif isinstance(inst, (DefGate)):  # pragma: no cover
            continue

        # Raise a targeted error when encountering a RESET.
        elif isinstance(inst, (Reset, ResetQubit)):
            raise UnsupportedQuilInstruction(RESET_ERROR)

        # Raise a general error when encountering an unconsidered type.
        else:
            raise UnsupportedQuilInstruction(
                f"Quil instruction {inst} of type {type(inst)} not currently supported in Cirq."
            )

    if len(kraus_model) > 0:  # pragma: no cover
        noise_model = kraus_noise_model_to_cirq(kraus_model, defined_gates)
        circuit = circuit.with_noise(noise_model)

    return circuit


def get_defined_gates(program: Program) -> Tuple[Dict, Dict]:
    """Get the gate definitions for the program. Will include the default SUPPORTED_GATES, in
    addition to any gates defined in the Quil

    Args:
        program: A pyquil program which may contain some DefGates.

    Returns:
        A dictionary mapping quil gate names to Cirq Gates
        A dictionary mapping quil gate names to callable parameter transformers
    """
    defined_gates = SUPPORTED_GATES.copy()
    parameter_transformers = PARAMETRIC_TRANSFORMERS.copy()
    for defgate in program.defined_gates:
        if defgate.parameters:
            defined_gates[defgate.name] = defgate_to_cirq(defgate)
            parameter_transformers[defgate.name] = lambda *args: {
                p.name: a for p, a in zip(defgate.parameters, args)
            }
        else:
            defined_gates[defgate.name] = MatrixGate(np.asarray(defgate.matrix, dtype=np.complex_))
    return defined_gates, parameter_transformers


def kraus_noise_model_to_cirq(
    kraus_noise_model: Dict[Tuple[QubitDesignator, ...], List[NDArray[np.complex_]]],
    defined_gates: Optional[Dict[QubitDesignator, Gate]] = None,
) -> InsertionNoiseModel:  # pragma: no cover
    """Construct a Cirq noise model from the provided Kraus operators.

    Args:
        kraus_noise_model: A dictionary where the keys are tuples of Quil gate names and qubit
        indices and the values are the Kraus representation of the noise channel.
        defined_gates: A dictionary mapping Quil gates to Cirq gates.
    Returns:
        A Cirq InsertionNoiseModel which applies the Kraus operators to the specified gates.
    Raises:
        Exception: If a QubitDesignator identifier is not an integer.
    """
    if defined_gates is None:
        # SUPPORTED_GATES values are all safe to use as `Gate`
        defined_gates = SUPPORTED_GATES  # type: ignore
    ops_added = {}
    for key, kraus_ops in kraus_noise_model.items():
        gate_name = key[0]

        try:
            qubit_indices = [int(q) for q in key[1:]]  # type: ignore
        except ValueError as e:
            raise Exception("Qubit identifier must be integers") from e
        qubits = [LineQubit(q) for q in qubit_indices]

        # defined_gates is not None by this point
        gate: Type[Gate] = defined_gates[gate_name]  # type: ignore
        target_op = OpIdentifier(gate, *qubits)

        insert_op = KrausChannel(kraus_ops, validate=True).on(*qubits)
        ops_added[target_op] = insert_op

    noise_model = InsertionNoiseModel(ops_added=ops_added, require_physical_tag=False)

    return noise_model


def quil_expression_to_sympy(expression: ParameterDesignator):
    """Convert a quil expression to a Sympy expression.

    Args:
        expression: A quil expression.

    Returns:
        The sympy form of the expression.

    Raises:
        ValueError: Connect convert unknown BinaryExp.
        ValueError: Unrecognized expression.
    """
    if type(expression) in {np.int_, np.float_, np.complex_, int, float, complex}:
        return expression
    elif isinstance(expression, Parameter):  # pragma: no cover
        return sympy.Symbol(expression.name)
    elif isinstance(expression, MemoryReference):
        return sympy.Symbol(expression.name + f"_{expression.offset}")
    elif isinstance(expression, Function):
        if expression.name == "SIN":  # pragma: no cover
            return sympy.sin(quil_expression_to_sympy(expression.expression))
        elif expression.name == "COS":
            return sympy.cos(quil_expression_to_sympy(expression.expression))
        elif expression.name == "SQRT":  # pragma: no cover
            return sympy.sqrt(quil_expression_to_sympy(expression.expression))
        elif expression.name == "EXP":
            return sympy.exp(quil_expression_to_sympy(expression.expression))
        elif expression.name == "CIS":  # pragma: no cover
            return sympy.exp(1j * quil_expression_to_sympy(expression.expression))
        else:  # pragma: no cover
            raise ValueError(f"Cannot convert unknown function: {expression}")

    elif isinstance(expression, BinaryExp):
        if isinstance(expression, Add):
            return quil_expression_to_sympy(expression.op1) + quil_expression_to_sympy(
                expression.op2
            )
        elif isinstance(expression, Sub):  # pragma: no cover
            return quil_expression_to_sympy(expression.op1) - quil_expression_to_sympy(
                expression.op2
            )
        elif isinstance(expression, Mul):
            return quil_expression_to_sympy(expression.op1) * quil_expression_to_sympy(
                expression.op2
            )
        elif isinstance(expression, Div):  # pragma: no cover
            return quil_expression_to_sympy(expression.op1) / quil_expression_to_sympy(
                expression.op2
            )
        elif isinstance(expression, Pow):  # pragma: no cover
            return quil_expression_to_sympy(expression.op1) ** quil_expression_to_sympy(
                expression.op2
            )
        else:  # pragma: no cover
            raise ValueError(f"Cannot convert unknown BinaryExp: {expression}")

    else:  # pragma: no cover
        raise ValueError(
            f"quil_expression_to_sympy failed to convert {expression} of type {type(expression)}"
        )


@cached_method
def defgate_to_cirq(defgate: DefGate):
    """Convert a Quil DefGate to a Cirq Gate class.

    For non-parametric gates, it's recommended to create `MatrixGate` object. This function is
    intended for the case of parametric gates.

    Args:
        defgate: A quil gate defintion.
    Returns:
        A subclass of `Gate` corresponding to the DefGate.
    """
    name = defgate.name
    matrix = defgate.matrix
    parameters = defgate.parameters
    dim = int(np.sqrt(matrix.shape[0]))
    if parameters:
        parameter_names = set(p.name for p in parameters)

        def constructor(self, **kwargs):
            for p, val in kwargs.items():
                assert p in parameter_names, f"{p} is not a known parameter"
                setattr(self, p, val)

        def unitary(self, *args):
            if parameters:
                parameter_map = {p: getattr(self, p.name) for p in parameters}
                return substitute_array(matrix, parameter_map)

    else:

        def constructor(self, **kwards: Any): ...

        def unitary(self, *args):  # pragma: no cover
            return matrix

    def circuit_diagram_info(
        self, args: CircuitDiagramInfoArgs
    ) -> CircuitDiagramInfo:  # pragma: no cover
        return CircuitDiagramInfo(wire_symbols=tuple(name for _ in range(dim)))

    def num_qubits(self):
        return defgate.num_args()

    gate = type(
        name,
        (Gate,),
        {
            "__init__": constructor,
            "_num_qubits_": num_qubits,
            "_unitary_": unitary,
            "_circuit_diagram_info_": circuit_diagram_info,
        },
    )
    return gate


def remove_gate_from_kraus(
    kraus_ops: List[NDArray[np.complex_]], gate_matrix: NDArray[np.complex_]
):  # pragma: no cover
    """Recover the kraus operators from a kraus composed with a gate.
    This function is the reverse of append_kraus_to_gate.

    Args:
        kraus_ops: A list of Kraus Operators.
        gate_matrix: The gate unitary.

    Returns:
        The noise channel without the gate unitary.
    """
    return [kju @ gate_matrix.conj().T for kju in kraus_ops]
