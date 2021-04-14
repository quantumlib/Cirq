import random
import time
from typing import cast, Optional

import numpy as np

import cirq
from cirq.contrib.quimb import MPSSimulator
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _run_normal_simulation(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    circuit: cirq.Circuit,
) -> cirq.StepResult:
    qubits = circuit.all_qubits()
    args = sim.create_act_on_args(0, qubits=list(qubits))
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    return results


def _run_join_args_simulation(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    circuit: cirq.Circuit,
) -> cirq.StepResult:
    # Initialize the ActOnArgs, one per qubit.
    qubits = circuit.all_qubits()
    args_map = {q: sim.create_act_on_args(0, qubits=[q]) for q in qubits}

    # Iterate
    for op in circuit.all_operations():
        # Go through the op's qubits and join any disparate ActOnArgs states
        # into a new combined state.
        op_args: Optional[TActOnArgs] = None
        for q in op.qubits:
            if op_args is None or q not in op_args.qubits:
                op_args = (
                    args_map[q] if op_args is None else cast(TActOnArgs, op_args).join(args_map[q])
                )
        assert op_args is not None

        # (Backfill the args map with the new value)
        for q in op_args.qubits:
            args_map[q] = op_args

        # Act on the args with the operation
        op_args.axes = tuple(op_args.qubit_map[q] for q in op.qubits)
        cirq.act_on(op, op_args)

    # Create the final ActOnArgs by joining everything together
    final_args = None
    for args in set(args_map.values()):
        final_args = args if final_args is None else cast(TActOnArgs, final_args).join(args)

    # Run an empty simulation to get the results
    *_, results = sim.simulate_moment_steps(cirq.Circuit(), None, qubits, final_args)
    return results


def _run_comparison(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    circuit: cirq.Circuit,
):
    t1 = time.perf_counter()
    results = _run_normal_simulation(sim, circuit)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    results1 = _run_join_args_simulation(sim, circuit)
    print(time.perf_counter() - t1)

    qubits = list(circuit.all_qubits())
    sam = results.sample(qubits, 100)
    sam1 = results1.sample(qubits, 100)
    sam = np.transpose(sam)
    sam1 = np.transpose(sam1)
    for i in range(len(qubits)):
        print(sam[i].mean())
        print(sam1[i].mean())


def _random_circuit(
    num_qubits: int,
    circuit_length: int,
    cnot_freq: float,
):
    circuit = cirq.Circuit()
    for i in range(circuit_length):
        for j in range(num_qubits):
            if random.random() < cnot_freq:
                circuit.append(cirq.CX(cirq.LineQubit(j), cirq.LineQubit(int(j + num_qubits / 2) % num_qubits)))
            else:
                circuit.append(cirq.H(cirq.LineQubit(j)) ** random.random())
    return circuit


def _clifford_circuit(
    num_qubits: int,
    circuit_length: int,
):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(num_qubits)
    for _ in range(circuit_length * num_qubits):
        x = np.random.randint(7)
        if x == 0:
            circuit.append(cirq.X(np.random.choice(qubits)))
        elif x == 1:
            circuit.append(cirq.Z(np.random.choice(qubits)))
        elif x == 2:
            circuit.append(cirq.Y(np.random.choice(qubits)))
        elif x == 3:
            circuit.append(cirq.S(np.random.choice(qubits)))
        elif x == 4:
            circuit.append(cirq.H(np.random.choice(qubits)))
        elif x == 5:
            circuit.append(cirq.CNOT(*np.random.choice(qubits, 2, replace=False)))
        elif x == 6:
            circuit.append(cirq.CZ(*np.random.choice(qubits, 2, replace=False)))
    return circuit


def main():
    print('***Run with Clifford simulator***')
    sim = cirq.CliffordSimulator()
    _run_comparison(sim, _clifford_circuit(num_qubits=20, circuit_length=10))

    print('***Run with MPS simulator***')
    sim = MPSSimulator()
    _run_comparison(sim, _random_circuit(num_qubits=30, circuit_length=100, cnot_freq=0.5))

    print('***Run with sparse simulator***')
    sim = cirq.Simulator()
    _run_comparison(sim, _random_circuit(num_qubits=22, circuit_length=10, cnot_freq=0.15))

    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator()
    _run_comparison(sim, _random_circuit(num_qubits=11, circuit_length=10, cnot_freq=0.15))


if __name__ == '__main__':
    main()
