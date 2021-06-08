"""Example application of the two-qubit gate compilation algorithm.

This script demonstrates construction and usage of the GateTabulation class.
This object is used to compile arbitrary two qubit gates using products of the
form

k_3 A k_2 A k_1 A k_0
or
k_2 A k_1 A k_0

where A is a fixed two-qubit gate and k_j are arbitrary single qubit gates.

The script generates the GateTabulation object associated with a base gate A
(sqrt(ISWAP) with some C-phase). It then generates a collection of 1000 randomly
generated 2-qubit gates and attempts to 'compile' them using the base gate.
Finally, it displays statistics on the process fidelity between the compiled
and desired gates.
"""

from time import time

import numpy as np
from matplotlib import pyplot as plt

import cirq
from cirq_google.optimizers.two_qubit_gates.gate_compilation import gate_product_tabulation
from cirq_google.optimizers.two_qubit_gates.math_utils import unitary_entanglement_fidelity
from cirq.testing import random_special_unitary


def main(samples: int = 1000, max_infidelity: float = 0.01):
    """Demonstration of the usage of the GateTabulation gate compiler.

    Args:
        samples: Number of random 2-qubit unitary samples to compile.
        max_infidelity: Maximum allowed infidelity between randomly generated
            gate and the gate compilation used to generate it. The example
            runtime scales as max_infidelity^{-2}.
    """
    # Define a base gate for compilation
    theta = np.pi / 4
    phi = np.pi / 24
    base = cirq.unitary(cirq.FSimGate(theta, phi))

    # The GateTabulation object is essentially a tabulation of many randomly
    # generated gate products (of the form A k A or A k A k A), along with their
    # associate KAK vectors. The parameter max_infidelity determines the
    # approximate "density" of the tabulated gates. Specifically, it bounds the
    # typical distance between an arbitrary two-qubit gate and the nearest
    # tabulated gate.
    start = time()
    tabulation = gate_product_tabulation(base, max_infidelity)

    print(tabulation.summary)
    print(f'Gate tabulation time : {time() - start} seconds.')

    # Generate many random two-qubit gates, then attempt to compile them using
    # the tabulation.
    unitaries = [random_special_unitary(4) for _ in range(samples)]

    infidelities = []
    failed_infidelities = []
    start = time()
    for target in unitaries:
        # result.actual_gate is the compiled gate product intended to match the
        # target. result.success denotes is the actual gate is expected to be
        # within the desired fidelity to the target. It can be False if the
        # base gate cannot "fill" the Weyl chamber using at most 3 products.
        # result.local_unitaries stores the local unitaries required to
        # compile the target unitary from the base unitary.
        result = tabulation.compile_two_qubit_gate(target)
        infidelity = 1 - unitary_entanglement_fidelity(target, result.actual_gate)
        if result.success:
            infidelities.append(infidelity)
        else:
            failed_infidelities.append(infidelity)  # coverage: ignore
    t_comp = time() - start

    print(f'Gate compilation time : {t_comp:.3f} seconds ({t_comp / samples:.4f} s per gate)')

    infidelities = np.array(infidelities)
    failed_infidelities = np.array(failed_infidelities)

    if np.size(failed_infidelities):
        # coverage: ignore
        print(f'Number of "failed" compilations: {np.size(failed_infidelities)}.')
        print(f'Maximum infidelity of "failed" compilation: {np.max(failed_infidelities)}')

    plt.figure()
    plt.hist(infidelities, bins=25, range=[0, max_infidelity * 1.1])
    ylim = plt.ylim()
    plt.plot([max_infidelity] * 2, ylim, '--', label='Maximum tabulation infidelity')
    plt.xlabel('Compiled gate infidelity vs target')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(f'Base FSim(theta={theta:.4f}, phi={phi:.4f})')

    plt.show()


if __name__ == '__main__':
    main()
