"""Tomography code for an arbitrary number of qubits allowing for
different pre-measurement rotations.
The code is designed to be modular with regards to data collection
so that occurs outside of the StateTomographyExperiment class.
"""
from typing import List, Tuple, Optional

import numpy as np
import sympy

import cirq
from cirq.experiments.qubit_characterizations import TomographyResult


class StateTomographyExperiment:
    """ Experiment to conduct state tomography.

    Generates data collection protocol for the state tomography experiment.
    Does the fitting of generated data to determine the density matrix.
    Attributes:
        rot_moment: Moment with parametrized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        mat: Matrix of coefficients for the system.  Each row is one equation
            corresponding to a rotation sequence and bit string outcome for
            that rotation sequence.  Each column corresponds to the coefficient
            on one term in the density matrix.
        num_qubits: Number of qubits to do tomography on.
    """

    def __init__(self,
                 qubits: List[cirq.Qid],
                 prerotations: Optional[List[Tuple[float, float]]] = None):
        """ Initializes the rotation protocol and matrix for system.

        Args:
            qubits: qubits to do the tomography on.
            prerotations: Tuples of (phase_exponent, exponent) parameters for
                gates to apply to the qubits before measurement. The actual
                rotation applied will be
                    cirq.PhasedXPowGate(
                        phase_exponent=phase_exponent, exponent=exponent)
                If none, we use [(0, 0), (0, 0.5), (0.5, 0.5)], which
                corresponds to rotation gates [I, X**0.5, Y**0.5]
        """
        if prerotations == None:
            prerotations = [(0, 0), (0, 0.5), (0.5, 0.5)]

        self.num_qubits = len(qubits)

        phase_exp_vals, exp_vals = zip(*prerotations)  # type: ignore

        ops: List[cirq.Operation] = []
        sweeps: List[cirq.Sweep] = []
        for i, qubit in enumerate(qubits):
            phase_exp = sympy.Symbol(f'phase_exp_{i}')
            exp = sympy.Symbol(f'exp_{i}')
            gate = cirq.PhasedXPowGate(phase_exponent=phase_exp, exponent=exp)
            ops.append(gate.on(qubit))
            sweeps.append(
                cirq.Points(phase_exp, phase_exp_vals) +
                cirq.Points(exp, exp_vals))

        self.rot_moment = cirq.Circuit.from_ops(ops)
        self.rot_sweep = cirq.Product(*sweeps)
        self.mat = self.make_state_tomography_matrix()

    def make_state_tomography_matrix(self) -> np.ndarray:
        """Gets the matrix used for solving the linear system of the tomography.

        Returns:
            A matrix of dimension ((number of rotations)**n * 2**n, 4**n)
            where each column corresponds to the coefficient of a term in the
            density matrix.  Each row is one equation corresponding to a
            rotation sequence and bit string outcome for that rotation sequence.
        """
        num_rots = len(self.rot_sweep)
        num_states = 2**self.num_qubits

        # Unitary matrices of each rotation circuit.
        Us = np.array([
            cirq.unitary(cirq.resolve_parameters(self.rot_moment, rots))
            for rots in self.rot_sweep
        ])
        mat = np.einsum('jkm,jkn->jkmn', Us, Us.conj())
        return mat.reshape((num_rots * num_states, num_states * num_states))

    def fit_density_matrix(
            self,
            counts: np.ndarray,
    ) -> 'TomographyResult':
        """Solves equation mat * rho = probs.

        Args:.
            counts:  A 2D array where each row contains measured counts
                of all n-qubit bitstrings for the corresponding pre-rotations
                in rot_sweep.  The order of the probabilities corresponds to
                to rot_sweep and the order of the bit strings corresponds to
                increasing integers up to 2**(num_qubits)-1

        Returns:
            TomographyResult with density matrix corresponding to solution of
            this system.
        """
        # normalize the input.
        probs = counts / np.sum(counts, axis=1)[:, np.newaxis]
        # use least squares to get solution.
        c, _, _, _ = np.linalg.lstsq(self.mat, np.asarray(probs).flat, rcond=-1)
        rho = c.reshape((2**self.num_qubits, 2**self.num_qubits))
        return TomographyResult(rho)


def state_tomography(
        sampler: cirq.Sampler,
        qubits: List[cirq.Qid],
        circuit: cirq.Circuit,
        repetitions: int = 1000,
        prerotations: List[Tuple[float, float]] = None,
) -> 'TomographyResult':
    """This performs n qubit tomography on a cirq circuit

    Follows https://web.physics.ucsb.edu/~martinisgroup/theses/Neeley2010b.pdf
    A.1. State Tomography.
    This is a high level interface for StateTomographyExperiment.
    Args:
        circuit: circuit to do the tomography on.
        qubits: qubits to do the tomography on.
        sampler: sampler to collect the data from.
        repetitions: number of times to sample each rotations
        prerotations: Tuples of (phase_exponent, exponent) parameters for gates
            to apply to the qubits before measurement. The actual rotation
            applied will be
                cirq.PhasedXPowGate(
                    phase_exponent=phase_exponent, exponent=exponent)
            If none, we use [(0, 0), (0, 0.5), (0.5, 0.5)], which corresponds to
            rotation gates [I, X**0.5, Y**0.5]
    Returns:
        Tomography result which contains the density matrix of the qubits
        determined by tomography.
    """
    if prerotations == None:
        prerotations = [(0, 0), (0, 0.5), (0.5, 0.5)]

    exp = StateTomographyExperiment(qubits, prerotations)
    rot_moment, rot_sweep = (exp.rot_moment, exp.rot_sweep)
    probs = get_state_tomography_data(sampler,
                                      qubits,
                                      circuit,
                                      rot_moment,
                                      rot_sweep,
                                      repetitions=repetitions)
    return exp.fit_density_matrix(probs)


def get_state_tomography_data(sampler: cirq.Sampler,
                              qubits: List[cirq.Qid],
                              circuit: cirq.Circuit,
                              rot_moment: cirq.Circuit,
                              rot_sweep: cirq.Sweep,
                              repetitions: int = 1000) -> np.ndarray:
    """Gets the data for each rotation string added to the circuit.

    For each sequence in prerotation_sequences gets the probability of all
    2**n bit strings.  Resulting matrix will have dimensions
    (len(rot_sweep)**n, 2**n).
    This is a default way to get data that can be replaced by the user if they
    have a more advanced protocol in mind.

    Args:
        sampler: sampler to collect the data from.
        qubits: qubits to do the tomography on.
        circuit: circuit to do the tomography on.
        rot_moment: moment with parametrized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        repetitions: number of times to sample each rotation sequence.

    Returns:
        2D array of probabilities, where first index is which pre-rotation was
        applied and second index is the qubit state.
    """
    results = sampler.run_sweep(circuit + rot_moment +
                                [cirq.measure(*qubits, key='z')],
                                params=rot_sweep,
                                repetitions=repetitions)

    all_probs = []
    for result in results:
        hist = result.histogram(key='z')
        probs = [hist[i] for i in range(2**len(qubits))]
        all_probs.append(np.array(probs) / repetitions)
    return np.array(all_probs)
