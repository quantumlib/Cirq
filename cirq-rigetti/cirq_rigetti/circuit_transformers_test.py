# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import List, Tuple
from unittest.mock import create_autospec

import numpy as np
from pyquil import Program
from pyquil.gates import CNOT, DECLARE, H, I, MEASURE, RX
from pyquil.quilbase import Pragma, Reset

import cirq
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti.deprecation import allow_deprecated_cirq_rigetti_use_in_tests


@allow_deprecated_cirq_rigetti_use_in_tests
def test_transform_cirq_circuit_to_pyquil_program(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace],
) -> None:
    """test that a user can transform a `cirq.Circuit` to a `pyquil.Program`
    functionally.
    """

    parametric_circuit, param_resolvers = parametric_circuit_with_params
    circuit = cirq.protocols.resolve_parameters(parametric_circuit, param_resolvers[1])
    program, _ = transformers.default(circuit=circuit)

    assert (
        RX(np.pi / 2, 0) in program.instructions
    ), "executable should contain an RX(pi) 0 instruction"
    assert DECLARE("m0") in program.instructions, "executable should declare a read out bit"
    assert (
        MEASURE(0, ("m0", 0)) in program.instructions
    ), "executable should measure the read out bit"


@allow_deprecated_cirq_rigetti_use_in_tests
def test_transform_cirq_circuit_to_pyquil_program_with_qubit_id_map(
    bell_circuit_with_qids: Tuple[cirq.Circuit, List[cirq.Qid]],
) -> None:
    """test that a user can transform a `cirq.Circuit` to a `pyquil.Program`
    functionally with explicit physical qubit address mapping.
    """

    bell_circuit, qubits = bell_circuit_with_qids

    qubit_id_map = {qubits[1]: "11", qubits[0]: "13"}
    transformer = transformers.build(qubit_id_map=qubit_id_map)
    program, _ = transformer(circuit=bell_circuit)

    assert H(13) in program.instructions, "bell circuit should include Hadamard"
    assert CNOT(13, 11) in program.instructions, "bell circuit should include CNOT"
    assert (
        DECLARE("m0", memory_size=2) in program.instructions
    ), "executable should declare a read out bit"
    assert (
        MEASURE(13, ("m0", 0)) in program.instructions
    ), "executable should measure the first qubit to the first read out bit"
    assert (
        MEASURE(11, ("m0", 1)) in program.instructions
    ), "executable should measure the second qubit to the second read out bit"


@allow_deprecated_cirq_rigetti_use_in_tests
def test_transform_with_post_transformation_hooks(
    bell_circuit_with_qids: Tuple[cirq.Circuit, List[cirq.Qid]],
) -> None:
    """test that a user can transform a `cirq.Circuit` to a `pyquil.Program`
    functionally with explicit physical qubit address mapping.
    """
    bell_circuit, qubits = bell_circuit_with_qids

    def reset_hook(program, measurement_id_map):
        program = Program(Reset()) + program
        return program, measurement_id_map

    reset_hook_spec = create_autospec(reset_hook, side_effect=reset_hook)

    pragma = Pragma('INTIAL_REWIRING', freeform_string='GREEDY')

    def rewire_hook(program, measurement_id_map):
        program = Program(pragma) + program
        return program, measurement_id_map

    rewire_hook_spec = create_autospec(rewire_hook, side_effect=rewire_hook)
    transformer = transformers.build(
        qubits=tuple(qubits), post_transformation_hooks=[reset_hook_spec, rewire_hook_spec]
    )
    program, _ = transformer(circuit=bell_circuit)

    assert 1 == reset_hook_spec.call_count
    assert Reset() in program.instructions, "hook should add reset"

    assert 1 == rewire_hook_spec.call_count
    assert pragma in program.instructions, "hook should add pragma"

    assert H(0) in program.instructions, "bell circuit should include Hadamard"
    assert CNOT(0, 1) in program.instructions, "bell circuit should include CNOT"
    assert (
        DECLARE("m0", memory_size=2) in program.instructions
    ), "executable should declare a read out bit"
    assert (
        MEASURE(0, ("m0", 0)) in program.instructions
    ), "executable should measure the first qubit to the first read out bit"
    assert (
        MEASURE(1, ("m0", 1)) in program.instructions
    ), "executable should measure the second qubit to the second read out bit"


@allow_deprecated_cirq_rigetti_use_in_tests
def test_transform_cirq_circuit_with_explicit_decompose(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace],
) -> None:
    """test that a user add a custom circuit decomposition function"""

    parametric_circuit, param_resolvers = parametric_circuit_with_params
    parametric_circuit.append(cirq.I(cirq.GridQubit(0, 0)))
    parametric_circuit.append(cirq.I(cirq.GridQubit(0, 1)))
    parametric_circuit.append(cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), key='m'))
    circuit = cirq.protocols.resolve_parameters(parametric_circuit, param_resolvers[1])

    def decompose_operation(operation: cirq.Operation) -> List[cirq.Operation]:
        operations = [operation]
        if isinstance(operation.gate, cirq.MeasurementGate) and operation.gate.num_qubits() == 1:
            operations.append(cirq.I(operation.qubits[0]))
        return operations

    program, _ = transformers.build(decompose_operation=decompose_operation)(circuit=circuit)

    assert (
        RX(np.pi / 2, 2) in program.instructions
    ), "executable should contain an RX(pi) 0 instruction"
    assert I(0) in program.instructions, "executable should contain an I(0) instruction"
    assert I(1) in program.instructions, "executable should contain an I(1) instruction"
    assert I(2) in program.instructions, "executable should contain an I(2) instruction"
    assert DECLARE("m0") in program.instructions, "executable should declare a read out bit"
    assert (
        MEASURE(0, ("m0", 0)) in program.instructions
    ), "executable should measure the read out bit"
