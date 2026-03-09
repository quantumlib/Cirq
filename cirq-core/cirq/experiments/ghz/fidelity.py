# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

import numpy as np

import cirq.circuits as circuits
import cirq.contrib.paulistring.pauli_string_measurement_with_readout_mitigation as psmrm
import cirq.ops as ops
import cirq.work as work


def int_to_stabilizer(which_stabilizer: int, qubits: list[ops.Qid]) -> ops.PauliString:
    """A mapping from the integers [0, ..., 2**num_qubits - 1] to GHZ stabilizers.

    First, `which_stabilizer` is converted to binary. The binary digits indicate whether
    the given basis stabilizer is present. The basis stabilizers, in order, are
    Z0*Z1, Z1*Z2, ..., Z(N-2)*Z(N-1), X0*X1*...*X(N-1).

    Args:
        which_stabilizer: The integer to convert to a stabilizer operator.
        qubits: The qubits in the GHZ state

    Returns:
        The stabilizer operator.
    """
    num_qubits = len(qubits)
    XXX = ops.X(qubits[0])
    for qubit in qubits[1:]:
        XXX *= ops.X(qubit)
    basis_ops = [ops.Z(qubits[i]) * ops.Z(qubits[i + 1]) for i in range(num_qubits - 1)] + [XXX]
    which_to_include = np.binary_repr(which_stabilizer, num_qubits)

    op_to_return = ops.I(qubits[0])
    for q in range(num_qubits):
        if which_to_include[-1 - q] == "1":
            op_to_return *= basis_ops[q]

    return op_to_return


def measure_ghz_fidelity(
    circuit: circuits.Circuit,
    num_z_type: int,
    num_x_type: int,
    rng: np.random.Generator,
    sampler: work.Sampler,
    pauli_repetitions: int = 10000,
    readout_repetitions: int = 10_000,
    num_random_bitstrings: int = 30,
) -> GHZFidelityResult:
    """Randomly sample z-type and x-type stabilizers of the GHZ state and measure them with and
        without readout error mitigation.

    Args:
        circuit: The circuit that prepares the GHZ state.
        num_z_type: The number of z-type stabilizers (all measured simultaneously)
        num_x_type: The number of x-type stabilizers
    """
    qubits = list(circuit.all_qubits())
    n_qubits = len(qubits)

    # pick random stabilizers
    z_type_ints = rng.choice(2 ** (n_qubits - 1), replace=False, size=num_z_type)
    x_type_ints = rng.choice(2 ** (len(qubits) - 1), replace=False, size=num_x_type) + 2 ** (
        len(qubits) - 1
    )

    z_type_paulis = [int_to_stabilizer(i, qubits) for i in z_type_ints]
    x_type_paulis = [int_to_stabilizer(i, qubits) for i in x_type_ints]

    paulis_to_measure = [z_type_paulis] + [[x] for x in x_type_paulis]
    circuits_to_pauli = {circuit.freeze(): paulis_to_measure}
    return GHZFidelityResult(
        psmrm.measure_pauli_strings(
            circuits_to_pauli,
            sampler,
            pauli_repetitions=pauli_repetitions,
            readout_repetitions=readout_repetitions,
            num_random_bitstrings=num_random_bitstrings,
            rng_or_seed=rng,
        )[0].results,
        num_z_type,
        num_x_type,
    )


class GHZFidelityResult:
    """A class for storing and analyzing the results of a GHZ fidelity benchmarking experiment."""

    def __init__(
        self, data: list[psmrm.PauliStringMeasurementResult], num_z_type: int, num_x_type: int
    ):
        self.data = data
        self.num_z_type = num_z_type
        self.num_x_type = num_x_type

    def compute_z_type_fidelity(self, mitigated: bool = True) -> tuple[float, float]:
        """Compute the z-type fidelity and statistical uncertainty.

        Args:
            mitigated: Whether to apply readout error mitigation.

        Returns:
            Return the average of the z-type stabilizers and the uncertainty of the average.
        """
        z_outcomes = [
            res.mitigated_expectation if mitigated else res.unmitigated_expectation
            for res in self.data[: self.num_z_type]
        ]
        return float(np.mean(z_outcomes)), float(np.std(z_outcomes) / np.sqrt(self.num_z_type))

    def compute_x_type_fidelity(self, mitigated: bool = True) -> tuple[float, float]:
        """Compute the x-type fidelity and statistical uncertainty.

        Args:
            mitigated: Whether to apply readout error mitigation.

        Returns:
            Return the average of the x-type stabilizers and the uncertainty of the average.
        """
        x_outcomes = [
            res.mitigated_expectation if mitigated else res.unmitigated_expectation
            for res in self.data[self.num_z_type :]
        ]
        assert len(x_outcomes) == self.num_x_type
        return float(np.mean(x_outcomes)), float(np.std(x_outcomes) / np.sqrt(self.num_x_type))

    def compute_fidelity(self, mitigated: bool = True) -> tuple[float, float]:
        """Compute the fidelity and statistical uncertainty.

        Args:
            mitigated: Whether to apply readout error mitigation.

        Returns:
            Return the average of the stabilizers and the uncertainty of the average.
        """
        z, dz = self.compute_z_type_fidelity(mitigated)
        x, dx = self.compute_x_type_fidelity(mitigated)
        return (x + z) / 2, np.sqrt(dx**2 + dz**2) / 2
