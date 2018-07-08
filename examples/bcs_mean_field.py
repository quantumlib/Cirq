"""  Quantum circuit to prepare the BCS ground states for superconductors/superfluids. Such states can be prepared by
applying pairwise Bogoliubov transformations on basis states with opposite spins and momenta, followed by the
fermionic Fourier transformations. In this simple example, we consider a 1D 4-site Hubbard model. The fermionic
quantum state is mapped that of a qubit ladder (two coupled chains) using the Jordan-Wigner transformation, the upper (
lower) chain represent spin-up (down) basis states.

The Bogoliubov transformation can be readily implemented by applying quantum gates on vertical pairs of qubits,
which takes the form |BCS⟩ = \prod_k (u_k + v_k c^\dag_{k,↑} c^\dag_{−k,↓}|vac⟩ where |vac⟩ is the vacuum state and
u_k^2 = (1+ ξ_k/(ξ_k^2+Δ_k^2)^{1/2})/2 and v_k^2 = (1 - ξ_k/(ξ_k^2+Δ_k^2)^{1/2})/2.

We use the fast fermionic Fourier transformation (FFFT) to implement the basis transformation from the momentum picture
to the position picture. This is an attempt to reduce the number of the gates that have to be calibrated in
experiments (compared to the Givens rotation approach); one only needs to calibrate a couple of two-qubit gates using
FFFT, i.e., the iSWAP gate and its square root √iSWAP. We use the single-qubit S gate to convert the iSWAP gate and
the √iSWAP gate to fermionic gates.

=== REFERENCE ===
F. Verstraete, J. I. Cirac, and J. I. Latorre, “Quantum circuits for strongly correlated quantum systems,”
Physical Review A 79, 032316 (2009).

Zhang Jiang, Kevin J. Sung, Kostyantyn Kechedzhi, Vadim N. Smelyanskiy, and Sergio Boixo
Phys. Rev. Applied 9, 044036 (2018).

=== EXAMPLE OUTPUT ===
Circuit for Bogoliubov transformation:
(0, 0): ───X───S───iSwap─────────Z^0.75───X───────────────────────────────────────────────────────────────────────────────────────────────────────────
                   │
(0, 1): ───────────┼──────────────────────────X───S───iSwap─────────Z^0.75───X────────────────────────────────────────────────────────────────────────
                   │                                  │
(0, 2): ───────────┼──────────────────────────────────┼──────────────────────────X───S───iSwap────────Z^0.75───X──────────────────────────────────────
                   │                                  │                                  │
(0, 3): ───────────┼──────────────────────────────────┼──────────────────────────────────┼─────────────────────────X───S───iSwap─────────Z^0.75───X───
                   │                                  │                                  │                                 │
(1, 0): ───────────iSwap^-1.68────────────────────────┼──────────────────────────────────┼─────────────────────────────────┼──────────────────────────
                                                      │                                  │                                 │
(1, 1): ──────────────────────────────────────────────iSwap^-1.47────────────────────────┼─────────────────────────────────┼──────────────────────────
                                                                                         │                                 │
(1, 2): ─────────────────────────────────────────────────────────────────────────────────iSwap^-1.0────────────────────────┼──────────────────────────
                                                                                                                           │
(1, 3): ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────iSwap^-1.47────────────────

Circuit for the inverse Fourier transformation on the spin-up states:
(0, 0): ───────────S^-1───iSwap───────S^-1──────────────────────────────S^-1───iSwap───────S^-1────────────────────────────────────────────
                          │                                                    │
(0, 1): ───iSwap───S^-1───iSwap^0.5─────────────────────────────iSwap───S^-1───iSwap^0.5────────────────────────────────────iSwap───S^-1───
           │                                                    │                                                           │
(0, 2): ───iSwap───S^-1───────────────S^-1───iSwap───────S^-1───iSwap───S^-1───────────────S──────S^-1───iSwap───────S^-1───iSwap───S^-1───
                                             │                                                           │
(0, 3): ─────────────────────────────────────iSwap^0.5───────────────────────────────────────────────────iSwap^0.5─────────────────────────

Circuit for the inverse Fourier transformation on the spin-down states:
(1, 0): ───────────S^-1───iSwap───────S^-1──────────────────────────────S^-1───iSwap───────S^-1────────────────────────────────────────────
                          │                                                    │
(1, 1): ───iSwap───S^-1───iSwap^0.5─────────────────────────────iSwap───S^-1───iSwap^0.5────────────────────────────────────iSwap───S^-1───
           │                                                    │                                                           │
(1, 2): ───iSwap───S^-1───────────────S^-1───iSwap───────S^-1───iSwap───S^-1───────────────S^-1───S^-1───iSwap───────S^-1───iSwap───S^-1───
                                             │                                                           │
(1, 3): ─────────────────────────────────────iSwap^0.5───────────────────────────────────────────────────iSwap^0.5─────────────────────────


"""

import cirq
import numpy as np
import scipy.optimize as sopt
from typing import List, Tuple


def main():

    # Number of sites in the Fermi-Hubbard model (2*n_site spin orbitals)
    n_site = 4
    # Number of fermions
    n_fermi = 4
    # Hopping strength between neighboring sites
    t = 1.
    # On-site interaction strength. It has to be negative (attractive) for the BCS theory to work.
    u = -6.
    # Calculate the superconducting gap and the angles in the Bogoliubov transformation
    delta, bog_theta = _bcs_parameters(n_site, n_fermi, u, t)
    # Initializing the qubits on a ladder
    q = _qubit_ladder(n_site)

    # The circuit for Bogoliubov transformation
    bog_circuit = cirq.Circuit()
    for i in range(n_site):
        bog = _bogoliubov_trans(q[0, i], q[1, i], bog_theta[i])
        bog_circuit.append(bog)
    print('Circuit for the Bogoliubov transformation:')
    print(bog_circuit)

    # The inverse fermionic Fourier transformation on the spin-up states
    fourier_circuit_spin_up = cirq.Circuit()
    print('Circuit for the inverse fermionic Fourier transformation on the spin-up states:')
    fourier_circuit_spin_up.append(_fermi_fourier_trans_inverse_4(q[0, :]))
    print(fourier_circuit_spin_up)

    # The inverse fermionic Fourier transformation on the spin-down states
    fourier_circuit_spin_down = cirq.Circuit()
    print('Circuit for the inverse fermionic Fourier transformation on the spin-down states:')
    fourier_circuit_spin_down.append(_fermi_fourier_trans_inverse_conjugate_4(q[1, :]))
    print(fourier_circuit_spin_down)


def _qubit_ladder(n_site: int) -> [cirq.QubitId]:

    """Initialize a qubit ladder to simulate the Hubbard model; the upper chain for spin-up basis states and the lower
    chain for spin-down basis states.

        Args:
             n_site: the length of the ladder (the number of sites in the Hubbard model).
    """

    quid_up = []
    quid_down = []
    for i in range(n_site):
        quid_up.append(cirq.devices.GridQubit(0, i))
        quid_down.append(cirq.devices.GridQubit(1, i))
    quid = np.asarray([quid_up, quid_down])
    quid = quid.reshape(2, n_site)

    return quid


def _fswap(p: cirq.QubitId, q: cirq.QubitId) -> [cirq.Operation]:

    """ Decompose the Fermionic SWAP gate into two single-qubit gates and one iSWAP gate.

        Args:
            p: the id of the first qubit
            q: the id of the second qubit
    """

    ops = [cirq.ISWAP(q, p), cirq.Z(p) ** 1.5, cirq.Z(q) ** 1.5]
    return ops


def _bogoliubov_trans(p: cirq.QubitId, q: cirq.QubitId, theta: float) -> [cirq.Operation]:

    """ The 2-mode Bogoliubov transformation is mapped to two-qubit operations. We use the identity X S^\dag X S X = Y
    X S^\dag Y S X = X to transform the Hamiltonian XY+YX to XX+YY type. The time evolution of the XX + YY
    Hamiltonian can be expressed as a power of the iSWAP gate.

        Args:
            p: the id of the first qubit
            q: the id of the second qubit
            theta: The rotational angle that specifies the Bogoliubov transformation, which is a function of
            the kinetic energy and the superconducting gap.
    """

    # The iSWAP gate corresponds to evolve under the Hamiltonian XX+YY for time -\pi/4.
    expo = -4.*theta/np.pi
    ops = [cirq.X(p),
           cirq.S(p),
           cirq.ISwapGate(exponent=expo).on(p, q),
           cirq.S(p) ** 1.5,
           cirq.X(p)]
    return ops


def _fermi_fourier_trans_2(p: cirq.QubitId, q: cirq.QubitId) -> [cirq.Operation]:

    """  The 2-mode fermionic Fourier transformation can be implemented straightforwardly by the √iSWAP gate.
    The √iSWAP gate can be readily implemented with the gmon qubits using the XX + YY Hamiltonian. The matrix
    representation of the 2-qubit fermionic Fourier transformation is:
    [1  0      0      0],
    [0  1/√2   1/√2   0],
    [0  1/√2  -1/√2   0],
    [0  0      0     -1]
    The square root of the iSWAP gate is:
    [1, 0, 0, 0],
    [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
    [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
    [0, 0, 0, 1]

        Args:
            p: the id of the first qubit
            q: the id of the second qubit
    """

    ops = [cirq.Z(p)**1.5,
           cirq.ISwapGate(exponent=0.5).on(q, p),
           cirq.Z(p)**1.5]
    return ops


def _fermi_fourier_trans_inverse_4(q: [cirq.QubitId]) -> [cirq.Operation]:

    """ The reverse fermionic Fourier transformation implemented on 4 qubits on a line, which maps the momentum
    picture to the position picture. Using the fast Fourier transformation algorithm, the circuit can be
    decomposed into 2-mode fermionic Fourier transformation, the fermionic SWAP gates, and single-qubit rotations.

        Args:
            q: the list of ids of the four qubits
    """

    ops = [_fswap(q[1], q[2]),
           _fermi_fourier_trans_2(q[0], q[1]),
           _fermi_fourier_trans_2(q[2], q[3]),
           _fswap(q[1], q[2]),
           _fermi_fourier_trans_2(q[0], q[1]),
           cirq.S(q[2]),
           _fermi_fourier_trans_2(q[2], q[3]),
           _fswap(q[1], q[2])]

    return ops


def _fermi_fourier_trans_inverse_conjugate_4(q: [cirq.QubitId]) -> [cirq.Operation]:

    """ We will need to map the momentum states in the reversed order for spin-down states to the position picture.
    This transformation can be simply implemented the complex conjugate of the former one. We only need to change the S
    gate to S* = S ** 3.

        Args:
            q: the list of ids of the four qubits
    """

    ops = [_fswap(q[1], q[2]),
           _fermi_fourier_trans_2(q[0], q[1]),
           _fermi_fourier_trans_2(q[2], q[3]),
           _fswap(q[1], q[2]),
           _fermi_fourier_trans_2(q[0], q[1]),
           cirq.S(q[2]) ** 3,
           _fermi_fourier_trans_2(q[2], q[3]),
           _fswap(q[1], q[2])]

    return ops


def _bcs_parameters(n_site: int, n_fermi: float, u: float, t: int) -> Tuple[float, List[float]]:

    """ Generate the parameters for the BCS ground state, i.e., the superconducting gap and the
    rotational angles for the Bogoliubov transformation.

         Args:
            n_site: the number of sites in the Hubbard model
            n_fermi: the number of fermions
            u: the interaction strength
            t: the tunneling strength
    """

    # The wave numbers satisfy the periodic boundary condition.
    wave_num = np.linspace(0, 1, n_site, endpoint=False)
    # The hopping energy as a function of wave numbers
    hop_erg = -2 * t * np.cos(2 * np.pi * wave_num)
    # Finding the Fermi energy
    fermi_erg = hop_erg[n_fermi // 2]
    # Set the Fermi energy to zero
    hop_erg = hop_erg - fermi_erg

    def _bcs_gap(delta: float) -> float:

        """ Defines the self-consistent equation for the BCS wavefunction.

            Args:
            delta: the superconducting gap
        """

        s = 0.
        for i in range(n_site):
            s += 1. / np.sqrt(hop_erg[i] ** 2 + delta ** 2)
        return 1 + s * u / (2 * n_site)

    # Superconducting gap
    delta = sopt.bisect(_bcs_gap, 0.01, 10000. * abs(u))
    # The parameter v in the Bogoliubov transformation, i.e., the amplitude of the double excitation state
    bcs_v = np.sqrt(0.5 * (1 - hop_erg / np.sqrt(hop_erg ** 2 + delta ** 2)))
    # The rotational angle in the Bogoliubov transformation.
    bog_theta = np.arcsin(bcs_v)

    return delta, bog_theta


if __name__ == "__main__":
    main()