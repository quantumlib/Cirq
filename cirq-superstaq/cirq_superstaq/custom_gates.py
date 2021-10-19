"""Miscellaneous custom gates that we encounter and want to explicitly define."""

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import cirq
import numpy as np

import cirq_superstaq


@cirq.value_equality(approximate=True)
class FermionicSWAPGate(cirq.Gate, cirq.ops.gate_features.InterchangeableQubitsGate):
    r"""The Fermionic SWAP gate, which performs the ZZ-interaction followed by a SWAP.

    Fermionic SWAPs are useful for applications like QAOA or Hamiltonian Simulation,
    particularly on linear- or low- connectivity devices. See https://arxiv.org/pdf/2004.14970.pdf
    for an application of Fermionic SWAP networks.

    The unitary for a Fermionic SWAP gate parametrized by ZZ-interaction angle :math:`\theta` is:

     .. math::

        \begin{bmatrix}
        1 & . & . & . \\
        . & . & e^{i \theta} & . \\
        . & e^{i \theta} & . & . \\
        . & . & . & 1 \\
        \end{bmatrix}

    where '.' means '0'.
    For :math:`\theta = 0`, the Fermionic SWAP gate is just an ordinary SWAP.

    Note that this gate is NOT the same as ``cirq.FSimGate``.
    """

    def __init__(self, theta: float) -> None:
        """
        Args:
            theta: ZZ-interaction angle in radians
        """
        self.theta = cirq.ops.fsim_gate._canonicalize(theta)  # between -pi and +pi

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, np.exp(1j * self.theta), 0],
                [0, np.exp(1j * self.theta), 0, 0],
                [0, 0, 0, 1],
            ]
        )

    def _value_equality_values_(self) -> Any:
        return self.theta

    def __str__(self) -> str:
        return f"FermionicSWAPGate({self.theta})"

    def __repr__(self) -> str:
        return f"cirq_superstaq.FermionicSWAPGate({self.theta})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        t = args.format_radians(self.theta)
        return cirq.CircuitDiagramInfo(wire_symbols=(f"FermionicSWAP({t})", f"FermionicSWAP({t})"))

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ["theta"])


class ZXPowGate(cirq.EigenGate, cirq.Gate):
    r"""The ZX-parity gate, possibly raised to a power.
    Per arxiv.org/pdf/1904.06560v3 eq. 135, the ZX**t gate implements the following unitary:
     .. math::
        e^{-\frac{i\pi}{2} t Z \otimes X} = \begin{bmatrix}
                                        c & -s & . & . \\
                                        -s & c & . & . \\
                                        . & . & c & s \\
                                        . & . & s & c \\
                                        \end{bmatrix}
    where '.' means '0' and :math:`c = \cos(\frac{\pi t}{2})`
    and :math:`s = i \sin(\frac{\pi t}{2})`.
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (
                0.0,
                np.array(
                    [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, -0.5], [0, 0, -0.5, 0.5]]
                ),
            ),
            (
                1.0,
                np.array(
                    [[0.5, -0.5, 0, 0], [-0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]
                ),
            ),
        ]

    def _eigen_shifts(self) -> List[float]:
        return [0, 1]

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=("Z", "X"), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self.exponent == 1:
            return "ZX"
        return f"ZX**{self._exponent!r}"

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return "cirq_superstaq.ZX"
            return f"(cirq_superstaq.ZX**{cirq._compat.proper_repr(self._exponent)})"
        return (
            f"cirq_superstaq.ZXPowGate(exponent={cirq._compat.proper_repr(self._exponent)},"
            f" global_shift={self._global_shift!r})"
        )


CR = ZX = ZXPowGate()  # standard CR is a full turn of ZX, i.e. exponent = 1


class AceCR(cirq.Gate):
    def __init__(self, polarity: str) -> None:
        assert polarity in ["+-", "-+"]
        self.polarity = polarity

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits: Tuple[cirq.LineQubit, cirq.LineQubit]) -> cirq.OP_TREE:
        yield cirq_superstaq.CR(*qubits) ** 0.25 if self.polarity == "+-" else cirq_superstaq.CR(
            *qubits
        ) ** -0.25
        yield cirq.X(qubits[0])
        yield cirq_superstaq.CR(*qubits) ** -0.25 if self.polarity == "+-" else cirq_superstaq.CR(
            *qubits
        ) ** 0.25

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=(f"AceCR{self.polarity}(Z side)", f"AceCR{self.polarity}(X side)")
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AceCR):
            return False
        return self.polarity == other.polarity

    def __hash__(self) -> int:
        return hash(self.polarity)

    def __repr__(self) -> str:
        return f"cirq_superstaq.AceCR('{self.polarity}')"

    def __str__(self) -> str:
        return f"AceCR{self.polarity}"

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ["polarity"])


AceCRMinusPlus = AceCR("-+")

AceCRPlusMinus = AceCR("+-")


class Barrier(cirq.ops.IdentityGate):
    """Barrier: temporal boundary restricting circuit compilation and pulse scheduling.
    Otherwise equivalent to the identity gate.
    """

    def _decompose_(self, qubits: Sequence["cirq.Qid"]) -> cirq.type_workarounds.NotImplementedType:
        return NotImplemented

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> str:
        indices_str = ",".join([f"{{{i}}}" for i in range(len(qubits))])
        format_str = f"barrier {indices_str};\n"
        return args.format(format_str, *qubits)

    def __str__(self) -> str:
        return f"Barrier({self.num_qubits()})"

    def __repr__(self) -> str:
        return f"cirq_superstaq.Barrier({self.num_qubits()})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        if args.use_unicode_characters:
            return ("│",) * self.num_qubits()
        return ("|",) * self.num_qubits()


@cirq.value_equality(approximate=True)
class ParallelGates(cirq.Gate, cirq.InterchangeableQubitsGate):
    """A single Gate combining a collection of concurrent Gate(s) acting on different qubits"""

    def __init__(self, *component_gates: cirq.Gate) -> None:
        """
        Args:
            component_gates: Gate(s) to be collected into single gate
        """
        if any(cirq.is_measurement(gate) for gate in component_gates):
            raise ValueError("ParallelGates cannot contain measurements")
        self.component_gates = component_gates

    def qubit_index_to_gate_and_index(self, index: int) -> Tuple[cirq.Gate, int]:
        for gate in self.component_gates:
            if gate.num_qubits() > index >= 0:
                return gate, index
            index -= gate.num_qubits()
        raise ValueError("index out of range")

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        indexed_gate, index_in_gate = self.qubit_index_to_gate_and_index(index)
        if indexed_gate.num_qubits() == 1:
            # find the first instance of the same gate
            first_instance = self.component_gates.index(indexed_gate)
            return sum(map(cirq.num_qubits, self.component_gates[:first_instance]))
        if isinstance(indexed_gate, cirq.InterchangeableQubitsGate):
            gate_key = indexed_gate.qubit_index_to_equivalence_group_key(index_in_gate)
            for i in range(index_in_gate):
                if gate_key == indexed_gate.qubit_index_to_equivalence_group_key(i):
                    return index - index_in_gate + i
        return index

    def _value_equality_values_(self) -> Tuple[cirq.Gate, ...]:
        return self.component_gates

    def _num_qubits_(self) -> int:
        return sum(map(cirq.num_qubits, self.component_gates))

    def _decompose_(self, qubits: Tuple[cirq.Qid, ...]) -> cirq.OP_TREE:
        """Decompose into each component gate"""
        for gate in self.component_gates:
            num_qubits = gate.num_qubits()
            yield gate(*qubits[:num_qubits])
            qubits = qubits[num_qubits:]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Generate a circuit diagram by connecting the wire symbols of each component gate.

        Symbols belonging to separate gates are differentiated via subscripts, with groups of
        symbols sharing the same subscript indicating multi-qubit operations.
        """
        wire_symbols_with_subscripts = []
        for i, gate in enumerate(self.component_gates):
            diagram_info = cirq.circuit_diagram_info(gate, args)
            full_wire_symbols = diagram_info._wire_symbols_including_formatted_exponent(
                args,
                preferred_exponent_index=cirq.num_qubits(gate) - 1,
            )

            index_str = f"_{i+1}"
            if args.use_unicode_characters:
                index_str = "".join(chr(ord("₁") + int(c)) for c in str(i))

            for base_symbol, full_symbol in zip(diagram_info.wire_symbols, full_wire_symbols):
                wire_symbols_with_subscripts.append(
                    full_symbol.replace(base_symbol, base_symbol + index_str)
                )
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols_with_subscripts)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ["component_gates"])

    @classmethod
    def _from_json_dict_(cls, component_gates: List[cirq.Gate], **kwargs: Any) -> Any:
        return cls(*component_gates)

    def __str__(self) -> str:
        component_gates_str = ", ".join(str(gate) for gate in self.component_gates)
        return f"ParallelGates({component_gates_str})"

    def __repr__(self) -> str:
        component_gates_repr = ", ".join(repr(gate) for gate in self.component_gates)
        return f"cirq_superstaq.ParallelGates({component_gates_repr})"


def custom_resolver(cirq_type: str) -> Union[Callable[..., cirq.Gate], None]:
    if cirq_type == "FermionicSWAPGate":
        return FermionicSWAPGate
    if cirq_type == "Barrier":
        return Barrier
    if cirq_type == "ZXPowGate":
        return ZXPowGate
    if cirq_type == "AceCR":
        return AceCR
    if cirq_type == "ParallelGates":
        return ParallelGates
    return None
