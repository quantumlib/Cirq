# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Using closed timelike curves for boolean satisfiability.

Assuming the existence of closed timelike curves, it becomes possible to solve
NP problems in polynomial time [1]. This is an example.

This example demonstrates a polynomial solution to the boolean satisfiability
problem. If closed timelike curves exist in nature, then they must be subject
to the restriction that their trace of the system's density matrix is the same
at the end of the experiment as it was at the beginning. Otherwise, history
would be inconsistent. This restriction typically completely determines the
initial state of the qubit in a closed timelike curve (CTC). This information
can then be used to maneuver the state of the system's non-CTC qubits toward a
desired goal. In particular, it allows for non-linearity that cannot be
attained with time-respecting qubits.

Here, a standard boolean satisfiability circuit is created to represent the
input problem. A target qubit is allocated, that is controlled by all solutions
of the problem. In a classical case, if, say, only one of the 2^N options was
a solution to the problem, then we would have to check each option until we
found it. Using quantum sampling gets us no closer (and in fact, further away):
each sample has a 1 in 2^N chance of succeeding, but it's probabilistic, so
requires more than 2^N samples to get to a reasonable probability that no
solution exists.

With CTC qubits however, non-linearity can be taken advantage of to quickly
guide the probability of each sample toward 0.5 if the expression is
satisfiable independent of how many solutions the problem has, while remaining
at 0 if the expression is unsatisfiable. Thus done, the resulting circuit can
be sampled K times. If the circuit is not solvable, all samples will be
negative. If the circuit is solvable, at least one sample will be positive,
with error probability 0.5^K.

[1] D. Bacon. Quantum Computational Complexity in the Presence of Closed
Timelike Curves (arXiv:quant-ph/0309189v2)

=== EXAMPLE OUTPUT ===
Expression: x0 & x1
Solutions: 1 of 4 possibilities
x0: ─────────X^0.5──────────@───────────────────────────────────────────
                            │
x1: ─────────X^0.5──────────@───────────────────────────────────────────
                            │
z_target: ──────────────────X───@───×───@───×───@───×───@───×───@───×───
                                │   │   │   │   │   │   │   │   │   │
zz_ctc0: ────BF(0.25+0j)────────X───×───┼───┼───┼───┼───┼───┼───┼───┼───
                                        │   │   │   │   │   │   │   │
zz_ctc1: ────BF(0.375+0j)───────────────X───×───┼───┼───┼───┼───┼───┼───
                                                │   │   │   │   │   │
zz_ctc2: ────BF(0.469+0j)───────────────────────X───×───┼───┼───┼───┼───
                                                        │   │   │   │
zz_ctc3: ────BF(0.498+0j)───────────────────────────────X───×───┼───┼───
                                                                │   │
zz_ctc4: ────BF(0.5+0j)─────────────────────────────────────────X───×───
Samples: 53/100 returned True
Satisfiable: True


Expression: x0 & ~x0
Solutions: 0 of 2 possibilities
x0: ─────────X^0.5──────────────────────────────────────

z_target: ───I──────────@───×───@───×───@───×───@───×───
                        │   │   │   │   │   │   │   │
zz_ctc0: ────BF(0+0j)───X───×───┼───┼───┼───┼───┼───┼───
                                │   │   │   │   │   │
zz_ctc1: ────BF(0+0j)───────────X───×───┼───┼───┼───┼───
                                        │   │   │   │
zz_ctc2: ────BF(0+0j)───────────────────X───×───┼───┼───
                                                │   │
zz_ctc3: ────BF(0+0j)───────────────────────────X───×───
Samples: 0/100 returned True
Satisfiable: False
"""

import itertools
import random

import numpy as np
import sympy.parsing.sympy_parser as sympy_parser

import cirq


def solve_with_closed_timelike_curves(boolean_str, expected):
    print()
    print('Expression: ' + boolean_str)
    # First use sympy to parse the expression.
    boolean_expr = sympy_parser.parse_expr(boolean_str)
    var_names = cirq.parameter_names(boolean_expr)
    n = len(var_names)

    # Now create the initial circuit from the paper: a qubit for each variable and set quantum state
    # to be the power set of combinations. In other words, a half X on each.
    circuit = cirq.Circuit()
    q_inputs = [cirq.NamedQubit(name) for name in var_names]
    circuit.append((cirq.X**0.5).on_each(*q_inputs))

    # Now, we need the initial unitary that sets up a `target` qubit with a CX whose controls are
    # exactly those that satisfy the boolean expression. Note that the paper suggests there is a
    # quadratic algorithm for this, though here we specify it as a single controlled X gate with the
    # matching control values. (So, here requiring exponential time to configure the gate).
    q_target = cirq.NamedQubit('z_target')
    solvers = []
    for binary_inputs in itertools.product([0, 1], repeat=n):
        subbed_expr = boolean_expr
        binary_inputs = list(binary_inputs)
        for var_name, binary_input in zip(var_names, binary_inputs):
            subbed_expr = subbed_expr.subs(var_name, binary_input)
        if bool(subbed_expr):
            solvers.append(binary_inputs)
    if solvers:
        cx = cirq.X(q_target).controlled_by(*q_inputs, control_values=cirq.SumOfProducts(solvers))
    else:
        cx = cirq.I(q_target)  # `SumOfProducts` does not accept an empty control set.
    circuit.append(cx)
    match_count = len(solvers)
    print(f'Solutions: {match_count} of {2**n} possibilities')

    # Now we append the CTC qubits and their interactions with the target qubit according to the
    # paper.
    initial_ctc_traces = []  # Store these for later check that CTC rules were not violated.
    n_ctc = n + 3  # Gets us very close to `target` being 50/50 if there's any solution.
    for i in range(n_ctc):
        # To find the initial trace of the next CTC qubit, we have to measure what the trace of
        # the target qubit would be at this point of circuit execution.
        dm = cirq.final_density_matrix(circuit).reshape((2,) * (n + 1 + i) * 2)
        target_trace = cirq.partial_trace(dm, [n])
        # The circuit as defined in the paper should only have identity and pauli-Z components
        # in the trace, thus it should be diagonal. Sanity check that this is the case.
        assert target_trace[0, 1] == target_trace[1, 0] == 0
        # Now we have to create and initialize the CTC qubit. Though the paper doesn't
        # thoroughly explain the algorithm for doing so, equation (1) could be used to solve it
        # analytically. However, here, we can cheat heuristically: given the interaction between
        # the target and the CTC is a CX with `target` as the control (leaving `target`
        # unchanged), followed by a SWAP, it is clear that the trace of CTC at the end of the
        # operation will be the trace of `target` currently. Since the CTC is required to have
        # the same trace at the start and the end of the operation, that means the initial trace
        # of the CTC must equal the current trace of `target`. Since `target's` trace is
        # diagonal, that corresponds to a bit flip with probability of the [1,1] element.
        q_ctc = cirq.NamedQubit(f'zz_ctc{i}')
        circuit.append(cirq.bit_flip(target_trace[1, 1]).on(q_ctc))
        # Sanity check that the traces are the same, and store it to check initial vs final
        # traces at the end of the algorithm.
        dm = cirq.final_density_matrix(circuit).reshape((2,) * (n + 2 + i) * 2)
        new_ctc_trace = cirq.partial_trace(dm, [n + i + 1])
        np.testing.assert_allclose(target_trace, new_ctc_trace, atol=1e-5)
        initial_ctc_traces.append(new_ctc_trace)
        # Finally add the interaction between `target` and the new CTC.
        circuit.append(cirq.CX(q_target, q_ctc))
        circuit.append(cirq.SWAP(q_target, q_ctc))

    # One can see from the CTC initializers in the circuit diagram that, assuming `target` and the
    # initial CTC traces are equal at each step, the target trends exponentially toward a 50/50
    # bit-flip.
    print(circuit)

    # Calculate the final density matrix, and check that traces were preserved on all CTC qubits.
    dm = cirq.final_density_matrix(circuit).reshape((2,) * (n + 1 + n_ctc) * 2)
    final_ctc_traces = [cirq.partial_trace(dm, [n + i + 1]) for i in range(n_ctc)]
    np.testing.assert_allclose(final_ctc_traces, initial_ctc_traces, atol=1e-5)

    # Finally, sample the density matrix of `target`. If *any* are True, then there is a solution.
    # If none are True, then the probability that there's a solution is roughly one in 2**100.
    results = cirq.sample_density_matrix(dm, [n], repetitions=100)
    n_passed = len([r for r in results if r[0]])
    print(f'Samples: {n_passed}/{len(results)} returned True')
    satisfiable = n_passed > 0
    print(f'Satisfiable: {satisfiable}')
    assert satisfiable == expected


def main(seed=None):
    """Simulate a boolean solver that uses closed timelike curves.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)
    satisfiable = ['x0', '~x0', 'x0 ^ x1', 'x0 & x1', 'x0 | x1', 'x0 & x1 & x2 & x3 & x4']
    unsatisfiable = ['x0 & ~x0', '(x0 | x1) & (x0 | ~x1) & (~x0 | x1) & (~x0 | ~x1)']
    for expr in satisfiable:
        solve_with_closed_timelike_curves(expr, expected=True)
    for expr in unsatisfiable:
        solve_with_closed_timelike_curves(expr, expected=False)


if __name__ == '__main__':
    main()
