# coding=utf-8
"""Quantum circuit to prepare the BCS ground states for
superconductors/superfluids. Such states can be prepared by
applying pairwise Bogoliubov transformations on basis states
with opposite spins and momenta, followed by the fermionic Fourier
transformations. In this simple example, we consider a 1D 4-site Hubbard model.
The fermionic quantum state is mapped that of a qubit ladder (two coupled
chains) using the Jordan-Wigner transformation, the upper (lower) chain
represent spin-up (down) basis states.

The Bogoliubov transformation can be readily implemented by
applying quantum gates on vertical pairs of qubits, which takes the form
|BCS⟩ = \prod_k (u_k + v_k c^\dag_{k,↑} c^\dag_{−k,↓}|vac⟩ where |vac⟩ is
the vacuum state and u_k^2 = (1+ ξ_k/(ξ_k^2+Δ_k^2)^{1/2})/2 and v_k^2
= (1 - ξ_k/(ξ_k^2+Δ_k^2)^{1/2})/2.

We use the fast fermionic Fourier transformation (FFFT) to implement the basis
transformation from the momentum picture to the position picture.
This is an attempt to reduce the number of the gates that have to be
calibrated in experiments (compared to the Givens rotation approach); one
only needs to calibrate a couple of two-qubit gates using FFFT, i.e.,
the iSWAP gate and its square root √iSWAP. We use the single-qubit S gate to
convert the iSWAP gate and the √iSWAP gate to fermionic gates.

=== REFERENCE ===
F. Verstraete, J. I. Cirac, and J. I. Latorre, “Quantum circuits for strongly
correlated quantum systems,” Physical Review A 79, 032316 (2009).

Zhang Jiang, Kevin J. Sung, Kostyantyn Kechedzhi, Vadim N. Smelyanskiy,
and Sergio Boixo Phys. Rev. Applied 9, 044036 (2018).

=== EXAMPLE OUTPUT ===
Quantum circuits to prepare the BCS mean field state.
Number of sites =  4
Number of fermions =  4
Tunneling strength =  1.0
On-site interaction strength =  -4.0
Superconducting gap =  1.1261371093950703

Circuit for the Bogoliubov transformation:
(0, 0)  (0, 1)  (0, 2)  (0, 3)  (1, 0)      (1, 1)      (1, 2)     (1, 3)
│       │       │       │       │           │           │          │
W(.25)  │       │       │       │           │           │          │
│       │       │       │       │           │           │          │
iSwap───┼───────┼───────┼───────iSwap^-1.83 │           │          │
│       │       │       │       │           │           │          │
W(.625) │       │       │       │           │           │          │
│       │       │       │       │           │           │          │
│       W(.25)  │       │       │           │           │          │
│       │       │       │       │           │           │          │
│       iSwap───┼───────┼───────┼───────────iSwap^-1.67 │          │
│       │       │       │       │           │           │          │
│       W(.625) │       │       │           │           │          │
│       │       │       │       │           │           │          │
│       │       W(.25)  │       │           │           │          │
│       │       │       │       │           │           │          │
│       │       iSwap───┼───────┼───────────┼───────────iSwap^-1.0 │
│       │       │       │       │           │           │          │
│       │       W(.625) │       │           │           │          │
│       │       │       │       │           │           │          │
│       │       │       W(.25)  │           │           │          │
│       │       │       │       │           │           │          │
│       │       │       iSwap───┼───────────┼───────────┼──────────iSwap^-1.67
│       │       │       │       │           │           │          │
│       │       │       W(.625) │           │           │          │
│       │       │       │       │           │           │          │


Circuit for the inverse fermionic Fourier transformation on the spin-up states:
(0, 0) (0, 1)    (0, 2) (0, 3)
│      │         │      │
S^-1   iSwap─────iSwap  │
│      │         │      │
│      S^-1      Z      │
│      │         │      │
iSwap──iSwap^0.5 │      │
│      │         │      │
Z      │         iSwap──iSwap^0.5
│      │         │      │
│      │         S^-1   │
│      │         │      │
│      iSwap─────iSwap  │
│      │         │      │
│      S^-1      S^-1   │
│      │         │      │
iSwap──iSwap^0.5 │      │
│      │         │      │
S^-1   │         │      │
│      │         │      │
│      │         iSwap──iSwap^0.5
│      │         │      │
│      │         S^-1   │
│      │         │      │
│      iSwap─────iSwap  │
│      │         │      │
│      S^-1      S^-1   │
│      │         │      │


Circuit for the inverse fermionic Fourier transformation on the spin-down
states:
(1, 0) (1, 1)    (1, 2) (1, 3)
│      │         │      │
S^-1   iSwap─────iSwap  │
│      │         │      │
│      S^-1      Z      │
│      │         │      │
iSwap──iSwap^0.5 │      │
│      │         │      │
Z      │         iSwap──iSwap^0.5
│      │         │      │
│      │         S^-1   │
│      │         │      │
│      iSwap─────iSwap  │
│      │         │      │
│      S^-1      S      │
│      │         │      │
iSwap──iSwap^0.5 │      │
│      │         │      │
S^-1   │         │      │
│      │         │      │
│      │         iSwap──iSwap^0.5
│      │         │      │
│      │         S^-1   │
│      │         │      │
│      iSwap─────iSwap  │
│      │         │      │
│      S^-1      S^-1   │
│      │         │      │


"""

from typing import List, Tuple, cast
import numpy as np
import scipy.optimize
import cirq


def main():

    # Number of sites in the Fermi-Hubbard model (2*n_site spin orbitals)
    n_site = 4
    # Number of fermions
    n_fermi = 4
    # Hopping strength between neighboring sites
    t = 1.
    # On-site interaction strength. It has to be negative (attractive) for the
    # BCS theory to work.
    u = -4.
    # Calculate the superconducting gap and the angles for BCS
    delta, bog_theta = bcs_parameters(n_site, n_fermi, u, t)
    # Initializing the qubits on a ladder
    upper_qubits = [cirq.GridQubit(0, i) for i in range(n_fermi)]
    lower_qubits = [cirq.GridQubit(1, i) for i in range(n_fermi)]

    print('Quantum circuits to prepare the BCS meanfield state.')
    print('Number of sites = ', n_site)
    print('Number of fermions = ', n_fermi)
    print('Tunneling strength = ', t)
    print('On-site interaction strength = ', u)
    print('Superconducting gap = ', delta, '\n')

    bog_circuit = cirq.Circuit.from_ops(
        bogoliubov_trans(upper_qubits[i], lower_qubits[i], bog_theta[i])
        for i in range(n_site))
    cirq.google.MergeRotations().optimize_circuit(bog_circuit)
    print('Circuit for the Bogoliubov transformation:')
    print(bog_circuit.to_text_diagram(transpose=True), '\n')

    # The inverse fermionic Fourier transformation on the spin-up states
    print(('Circuit for the inverse fermionic Fourier transformation on the '
           'spin-up states:'))
    fourier_circuit_spin_up = cirq.Circuit.from_ops(
        fermi_fourier_trans_inverse_4(upper_qubits),
        strategy=cirq.InsertStrategy.EARLIEST)
    cirq.google.MergeRotations().optimize_circuit(fourier_circuit_spin_up)
    print(fourier_circuit_spin_up.to_text_diagram(transpose=True), '\n')

    # The inverse fermionic Fourier transformation on the spin-down states
    print(('Circuit for the inverse fermionic Fourier transformation on the '
           'spin-down states:'))
    fourier_circuit_spin_down = cirq.Circuit.from_ops(
        fermi_fourier_trans_inverse_conjugate_4(lower_qubits),
        strategy=cirq.InsertStrategy.EARLIEST)
    cirq.google.MergeRotations().optimize_circuit(fourier_circuit_spin_down)
    print(fourier_circuit_spin_down.to_text_diagram(transpose=True))


def fswap(p, q):
    """Decompose the Fermionic SWAP gate into two single-qubit gates and
    one iSWAP gate.

    Args:
        p: the id of the first qubit
        q: the id of the second qubit
    """

    yield cirq.ISWAP(q, p), cirq.Z(p) ** 1.5
    yield cirq.Z(q) ** 1.5


def bogoliubov_trans(p, q, theta):
    """The 2-mode Bogoliubov transformation is mapped to two-qubit operations.
     We use the identity X S^\dag X S X = Y X S^\dag Y S X = X to transform
     the Hamiltonian XY+YX to XX+YY type. The time evolution of the XX + YY
     Hamiltonian can be expressed as a power of the iSWAP gate.

    Args:
        p: the first qubit
        q: the second qubit
        theta: The rotational angle that specifies the Bogoliubov
        transformation, which is a function of the kinetic energy and
        the superconducting gap.
    """

    # The iSWAP gate corresponds to evolve under the Hamiltonian XX+YY for
    # time -pi/4.
    expo = -4 * theta / np.pi

    yield cirq.X(p)
    yield cirq.S(p)
    yield cirq.ISwapGate(exponent=expo).on(p, q)
    yield cirq.S(p) ** 1.5
    yield cirq.X(p)


def fermi_fourier_trans_2(p, q):
    """The 2-mode fermionic Fourier transformation can be implemented
    straightforwardly by the √iSWAP gate. The √iSWAP gate can be readily
    implemented with the gmon qubits using the XX + YY Hamiltonian. The matrix
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
        p: the first qubit
        q: the second qubit
    """

    yield cirq.Z(p)**1.5
    yield cirq.ISwapGate(exponent=0.5).on(q, p)
    yield cirq.Z(p)**1.5


def fermi_fourier_trans_inverse_4(qubits):
    """The reverse fermionic Fourier transformation implemented on 4 qubits
    on a line, which maps the momentum picture to the position picture.
    Using the fast Fourier transformation algorithm, the circuit can be
    decomposed into 2-mode fermionic Fourier transformation, the fermionic
    SWAP gates, and single-qubit rotations.

    Args:
        qubits: list of four qubits
    """

    yield fswap(qubits[1], qubits[2]),
    yield fermi_fourier_trans_2(qubits[0], qubits[1])
    yield fermi_fourier_trans_2(qubits[2], qubits[3])
    yield fswap(qubits[1], qubits[2])
    yield fermi_fourier_trans_2(qubits[0], qubits[1])
    yield cirq.S(qubits[2])
    yield fermi_fourier_trans_2(qubits[2], qubits[3])
    yield fswap(qubits[1], qubits[2])


def fermi_fourier_trans_inverse_conjugate_4(qubits):
    """We will need to map the momentum states in the reversed order for
    spin-down states to the position picture. This transformation can be
    simply implemented the complex conjugate of the former one. We only
    need to change the S gate to S* = S ** 3.

    Args:
        qubits: list of four qubits
    """

    yield fswap(qubits[1], qubits[2]),
    yield fermi_fourier_trans_2(qubits[0], qubits[1])
    yield fermi_fourier_trans_2(qubits[2], qubits[3])
    yield fswap(qubits[1], qubits[2])
    yield fermi_fourier_trans_2(qubits[0], qubits[1])
    yield cirq.S(qubits[2]) ** 3
    yield fermi_fourier_trans_2(qubits[2], qubits[3])
    yield fswap(qubits[1], qubits[2])


def bcs_parameters(n_site, n_fermi, u, t) :
    """Generate the parameters for the BCS ground state, i.e., the
    superconducting gap and the rotational angles in the Bogoliubov
    transformation.

     Args:
        n_site: the number of sites in the Hubbard model
        n_fermi: the number of fermions
        u: the interaction strength
        t: the tunneling strength

    Returns:
        float delta, List[float] bog_theta
    """

    # The wave numbers satisfy the periodic boundary condition.
    wave_num = np.linspace(0, 1, n_site, endpoint=False)
    # The hopping energy as a function of wave numbers
    hop_erg = -2 * t * np.cos(2 * np.pi * wave_num)
    # Finding the Fermi energy
    fermi_erg = hop_erg[n_fermi // 2]
    # Set the Fermi energy to zero
    hop_erg = hop_erg - fermi_erg

    def _bcs_gap(x):
        """Defines the self-consistent equation for the BCS wavefunction.

        Args:
            x: the superconducting gap
        """

        s = 0.
        for i in range(n_site):
            s += 1. / np.sqrt(hop_erg[i] ** 2 + x ** 2)
        return 1 + s * u / (2 * n_site)

    # Superconducting gap
    delta = scipy.optimize.bisect(_bcs_gap, 0.01, 10000. * abs(u))
    delta = cast(float, delta)
    # The amplitude of the double excitation state
    bcs_v = np.sqrt(0.5 * (1 - hop_erg / np.sqrt(hop_erg ** 2 + delta ** 2)))
    # The rotational angle in the Bogoliubov transformation.
    bog_theta = np.arcsin(bcs_v)

    return delta, bog_theta


if __name__ == "__main__":
    main()
