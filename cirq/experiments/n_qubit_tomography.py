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
"""Tomography code for an arbitrary number of qubits allowing for
different pre-measurement rotations.

The code is designed to be modular with regards to data collection
so that occurs outside of the StateTomographyExperiment class.
"""
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import sympy

from cirq import circuits, ops, protocols, study
from cirq.experiments.qubit_characterizations import TomographyResult

if TYPE_CHECKING:
    import cirq


class StateTomographyExperiment:
    """Experiment to conduct state tomography.

    Generates data collection protocol for the state tomography experiment.
    Does the fitting of generated data to determine the density matrix.

    Attributes:
        rot_circuit: Circuit with parameterized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        mat: Matrix of coefficients for the system.  Each row is one equation
            corresponding to a rotation sequence and bit string outcome for
            that rotation sequence.  Each column corresponds to the coefficient
            on one term in the density matrix.
        num_qubits: Number of qubits to do tomography on.
    """

    def __init__(
        self,
        qubits: Sequence['cirq.Qid'],
        prerotations: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        """Initializes the rotation protocol and matrix for system.

        Args:
            qubits: Qubits to do the tomography on.
            prerotations: Tuples of (phase_exponent, exponent) parameters for
                gates to apply to the qubits before measurement. The actual
                rotation applied will be `cirq.PhasedXPowGate` with the
                specified values of phase_exponent and exponent. If None,
                we use [(0, 0), (0, 0.5), (0.5, 0.5)], which corresponds
                to rotation gates [I, X**0.5, Y**0.5].
        """
        if prerotations is None:
            prerotations = [(0, 0), (0, 0.5), (0.5, 0.5)]
        self.num_qubits = len(qubits)

        phase_exp_vals, exp_vals = zip(*prerotations)

        operations: List['cirq.Operation'] = []
        sweeps: List['cirq.Sweep'] = []
        for i, qubit in enumerate(qubits):
            phase_exp = sympy.Symbol(f'phase_exp_{i}')
            exp = sympy.Symbol(f'exp_{i}')
            gate = ops.PhasedXPowGate(phase_exponent=phase_exp, exponent=exp)
            operations.append(gate.on(qubit))
            sweeps.append(study.Points(phase_exp, phase_exp_vals) + study.Points(exp, exp_vals))

        self.rot_circuit = circuits.Circuit(operations)
        self.rot_sweep = study.Product(*sweeps)
        self.mat = self._make_state_tomography_matrix(qubits)

    def _make_state_tomography_matrix(
        self,
        qubits: Sequence['cirq.Qid'],
    ) -> np.ndarray:
        """Gets the matrix used for solving the linear system of the tomography.

        Args:
            qubits: Qubits to do the tomography on.

        Returns:
            A matrix of dimension ((number of rotations)**n * 2**n, 4**n)
            where each column corresponds to the coefficient of a term in the
            density matrix.  Each row is one equation corresponding to a
            rotation sequence and bit string outcome for that rotation sequence.
        """
        num_rots = len(self.rot_sweep)
        num_states = 2 ** self.num_qubits

        # Unitary matrices of each rotation circuit.
        unitaries = np.array(
            [
                protocols.resolve_parameters(self.rot_circuit, rots).unitary(qubit_order=qubits)
                for rots in self.rot_sweep
            ]
        )
        mat = np.einsum('jkm,jkn->jkmn', unitaries, unitaries.conj())
        return mat.reshape((num_rots * num_states, num_states * num_states))

    def fit_density_matrix(self, counts: np.ndarray) -> TomographyResult:
        """Solves equation mat * rho = probs.

        Args:
            counts: A 2D array where each row contains measured counts
                of all n-qubit bitstrings for the corresponding pre-rotations
                in `rot_sweep`.  The order of the probabilities corresponds to
                to `rot_sweep` and the order of the bit strings corresponds to
                increasing integers up to 2**(num_qubits)-1.

        Returns:
            `TomographyResult` with density matrix corresponding to solution of
            this system.
        """
        # normalize the input.
        probs = counts / np.sum(counts, axis=1)[:, np.newaxis]
        # use least squares to get solution.
        c, _, _, _ = np.linalg.lstsq(self.mat, np.asarray(probs).flat, rcond=-1)
        rho = c.reshape((2 ** self.num_qubits, 2 ** self.num_qubits))
        return TomographyResult(rho)


def state_tomography(
    sampler: 'cirq.Sampler',
    qubits: Sequence['cirq.Qid'],
    circuit: 'cirq.Circuit',
    repetitions: int = 1000,
    prerotations: Sequence[Tuple[float, float]] = None,
) -> TomographyResult:
    """This performs n qubit tomography on a cirq circuit

    Follows https://web.physics.ucsb.edu/~martinisgroup/theses/Neeley2010b.pdf
    A.1. State Tomography.
    This is a high level interface for StateTomographyExperiment.

    Args:
        circuit: Circuit to do the tomography on.
        qubits: Qubits to do the tomography on.
        sampler: Sampler to collect the data from.
        repetitions: Number of times to sample each rotation.
        prerotations: Tuples of (phase_exponent, exponent) parameters for gates
            to apply to the qubits before measurement. The actual rotation
            applied will be `cirq.PhasedXPowGate` with the specified values
            of phase_exponent and exponent. If None, we use [(0, 0), (0, 0.5),
            (0.5, 0.5)], which corresponds to rotation gates
            [I, X**0.5, Y**0.5].

    Returns:
        `TomographyResult` which contains the density matrix of the qubits
        determined by tomography.
    """
    exp = StateTomographyExperiment(qubits, prerotations)
    probs = get_state_tomography_data(
        sampler, qubits, circuit, exp.rot_circuit, exp.rot_sweep, repetitions=repetitions
    )
    return exp.fit_density_matrix(probs)


def get_state_tomography_data(
    sampler: 'cirq.Sampler',
    qubits: Sequence['cirq.Qid'],
    circuit: 'cirq.Circuit',
    rot_circuit: 'cirq.Circuit',
    rot_sweep: 'cirq.Sweep',
    repetitions: int = 1000,
) -> np.ndarray:
    """Gets the data for each rotation string added to the circuit.

    For each sequence in prerotation_sequences gets the probability of all
    2**n bit strings.  Resulting matrix will have dimensions
    (len(rot_sweep)**n, 2**n).
    This is a default way to get data that can be replaced by the user if they
    have a more advanced protocol in mind.

    Args:
        sampler: Sampler to collect the data from.
        qubits: Qubits to do the tomography on.
        circuit: Circuit to do the tomography on.
        rot_circuit: Circuit with parameterized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        repetitions: Number of times to sample each rotation sequence.

    Returns:
        2D array of probabilities, where first index is which pre-rotation was
        applied and second index is the qubit state.
    """
    results = sampler.run_sweep(
        circuit + rot_circuit + [ops.measure(*qubits, key='z')],
        params=rot_sweep,
        repetitions=repetitions,
    )

    all_probs = []
    for result in results:
        hist = result.histogram(key='z')
        probs = [hist[i] for i in range(2 ** len(qubits))]
        all_probs.append(np.array(probs) / repetitions)
    return np.array(all_probs)
