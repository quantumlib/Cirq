"""Demonstrates Shor's algorithm.

Shor's algorithm [1] is a quantum algorithm for integer factorization. Given
a composite integer n, it finds its non-trivial factor d, i.e. a factor other
than 1 or n.

The algorithm consists of two parts: quantum order-finding subroutine and
classical probabilistic reduction of the factoring problem to the order-
finding problem. Given two positive integers x and n, the order-finding
problem asks for the smallest positive integer r such that x^r mod n == 1.

The classical reduction algorithm first handles two corner cases which do
not rely on quantum computation: when n is even or a prime power. For other
n, the algorithm first draws a random x uniformly from 2..n-1 and then uses
the quantum order-finding subroutine to compute the order r of x modulo n,
i.e. it finds the smallest positive integer r such that x^r == 1 mod n. Now,
if r is even, then y = x^(r/2) is a solution to the equation

    y^2 == 1 mod n.                                                    (*)

It's easy to see that in this case gcd(y - 1, n) or gcd(y + 1, n) divides n.
If in addition y is a non-trivial solution, i.e. if it is not equal to -1,
then gcd(y - 1, n) or gcd(y + 1, n) is a non-trivial factor of n (note that
y cannot be 1). If r is odd or if y is a trivial solution of (*), then the
algorithm is repeated for a different random x.

It turns out [1] that the probability of r being even and y = x^(r/2) being
a non-trivial solution of equation (*) is at least 1 - 1/2^(k - 1) where k
is the number of distinct prime factors of n. Since the case k = 1 has been
handled by the classical part, we have k >= 2 and the success probability of
a single attempt is at least 1/2.

The subroutine for finding the order r of a number x modulo n consists of two
steps. In the first step, Quantum Phase Estimation is applied to a unitary such
as

    U|y⟩ = |xy mod n⟩  0 <= y < n
    U|y⟩ = |y⟩         n =< y

whose eigenvalues are s/r for s = 0, 1, ..., r - 1. In the second step, the
classical continued fractions algorithm is used to recover r from s/r. Note
that when gcd(s, r) > 1 then an incorrect r is found. This can be detected
by verifying that r is indeed the order of x. If it is not, Quantum Phase
Estimation algorithm is retried.

[1]: https://arxiv.org/abs/quant-ph/9508027
"""

import argparse
import fractions
import math
import random

from typing import Callable, List, Optional, Sequence, Union

import cirq
import sympy

parser = argparse.ArgumentParser(description='Factorization demo.')
parser.add_argument('n', type=int, help='composite integer to factor')
parser.add_argument('--order_finder',
                    type=str,
                    choices=('naive', 'quantum'),
                    default='naive',
                    help=('order finder to use; must be either "naive" '
                          'for a naive classical algorithm or "quantum" '
                          'for a quantum circuit; note that in practice '
                          '"quantum" is substantially slower since it '
                          'incurs the overhead of classical simulation.'))


def naive_order_finder(x: int, n: int) -> Optional[int]:
    """Computes smallest positive r such that x^r mod n == 1.

    Args:
        x: integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: modulus of the multiplicative group.

    Returns:
        Smallest positive integer r such that x^r == 1 mod n.
        Always succeeds (and hence never returns None).

    Raises:
        ValueError when x is 1 or not an element of the multiplicative
        group of integers modulo n.
    """
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r


class ModularExp(cirq.ArithmeticOperation):
    """Quantum modular exponentiation.

    This class represents the unitary which multiplies a given exponent of
    x into the ancilla register. More precisely, it represents the unitary V
    which computes modular exponentiation x^e mod n:

        V|y⟩|e⟩ = |y * x^e mod n⟩ |e⟩      0 <= y < n
        V|y⟩|e⟩ = |y⟩ |e⟩                  n <= y

    where y is the ancilla register, e is the exponent register and x and n are
    non-negative integer constants. Consequently,

        V|y⟩|e⟩ = (U^e|r⟩)|e⟩

    where U is the unitary defined as

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    in the header of this file.

    Quantum order finding algorithm (which is the quantum part of the Shor's
    algorithm) uses quantum modular exponentiation together with the Quantum
    Phase Estimation to compute the order of x modulo n.
    """

    def __init__(self, ancilla_register: Sequence[cirq.Qid],
                 exponent_register: Sequence[cirq.Qid], x: int, n: int) -> None:
        self.ancilla_register = ancilla_register
        self.exponent_register = exponent_register
        self.x = x
        self.n = n

    def registers(self) -> Sequence[Sequence[cirq.Qid]]:
        return self.ancilla_register, self.exponent_register

    def with_registers(
            self,
            *new_registers: Union[int, Sequence['cirq.Qid']],
    ) -> cirq.ArithmeticOperation:
        # coverage: ignore
        assert len(new_registers) == 2
        assert isinstance(new_registers[0], Sequence)
        assert isinstance(new_registers[1], Sequence)
        return ModularExp(new_registers[0], new_registers[1], self.x, self.n)

    def apply(self, *register_values: int) -> int:
        assert len(register_values) == 2
        result, exponent = register_values
        if result >= self.n:
            return result
        return (result * self.x**exponent) % self.n

    def _circuit_diagram_info_(
            self,
            args: cirq.CircuitDiagramInfoArgs,
    ) -> cirq.CircuitDiagramInfo:
        assert args.known_qubits is not None
        wire_symbols: List[str] = []
        a, e = 0, 0
        for qubit in args.known_qubits:
            if qubit in self.ancilla_register:
                if a == 0:
                    wire_symbols.append(f'ModularExp(a*{self.x}^e % {self.n})')
                else:
                    wire_symbols.append('a' + str(a))
                a += 1
            if qubit in self.exponent_register:
                wire_symbols.append('e' + str(e))
                e += 1
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))


def bitlength(n: int) -> int:
    """Returns minimum number of bits needed to specify a positive integer."""
    return int(math.ceil(math.log(n) / math.log(2)))


def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:
    """Returns quantum circuit which computes the order of x modulo n.

    The circuit uses Quantum Phase Estimation to compute an eigenvalue of
    the unitary

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    discussed in the header of this file. The circuit uses two registers:
    the ancilla register which is acted on by U and the exponent register
    from which an eigenvalue is read out after measurement at the end. The
    circuit consists of three steps:

    1. Initialization of the ancilla register to |0..01⟩ and the exponent
       register to a superposition state.
    2. Multiple controlled-U^2^j operations implemented efficiently using
       modular exponentiation.
    3. Inverse Quantum Fourier Transform to kick an eigenvalue to the
       exponent register.

    Args:
        x: positive integer whose order modulo n is to be found
        n: modulus relative to which the order of x is to be found

    Returns:
        Quantum circuit for finding the order of x modulo n
    """
    L = bitlength(n)
    ancilla_register = cirq.LineQubit.range(L)
    exponent_register = cirq.LineQubit.range(L, 3 * L + 3)
    return cirq.Circuit(
        cirq.X(ancilla_register[L - 1]),
        cirq.H.on_each(*exponent_register),
        ModularExp(ancilla_register, exponent_register, x, n),
        cirq.QFT(*exponent_register, inverse=True),
        cirq.measure(*exponent_register, key='exponent'),
    )


def read_eigenphase(result: cirq.TrialResult) -> float:
    """Interprets the output of the order finding circuit.

    Specifically, it returns s/r such that exp(2πis/r) is an eigenvalue
    of the unitary

        U|y⟩ = |xy mod n⟩  0 <= y < n
        U|y⟩ = |y⟩         n <= y

    described in the header of this file.

    Args:
        result: trial result obtained by sampling the output of the
            circuit built by make_order_finding_circuit

    Returns:
        s/r where r is the order of x modulo n and s is in [0..r-1].
    """
    exponent_as_integer = result.data['exponent'][0]
    exponent_num_bits = result.measurements['exponent'].shape[1]
    return float(exponent_as_integer / 2**exponent_num_bits)


def quantum_order_finder(x: int, n: int) -> Optional[int]:
    """Computes smallest positive r such that x^r mod n == 1.

    Args:
        x: integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: modulus of the multiplicative group.

    Returns:
        Smallest positive integer r such that x^r == 1 mod n or None if the
        algorithm failed. The algorithm fails when the result of the Quantum
        Phase Estimation is inaccurate, zero or a reducible fraction.

    Raises:
        ValueError when x is 1 or not an element of the multiplicative
        group of integers modulo n.
    """
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    circuit = make_order_finding_circuit(x, n)
    result = cirq.sample(circuit)
    eigenphase = read_eigenphase(result)
    f = fractions.Fraction.from_float(eigenphase).limit_denominator(n)
    if f.numerator == 0:
        return None  # coverage: ignore
    r = f.denominator
    if x**r % n != 1:
        return None  # coverage: ignore
    return r


def find_factor_of_prime_power(n: int) -> Optional[int]:
    """Returns non-trivial factor of n if n is a prime power, else None."""
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1**k == n:
            return c1
        c2 = math.ceil(c)
        if c2**k == n:
            return c2
    return None


def find_factor(n: int,
                order_finder: Callable[[int, int], Optional[int]],
                max_attempts: int = 30) -> Optional[int]:
    """Returns a non-trivial factor of composite integer n.

    Args:
        n: integer to factorize,
        order_finder: function for finding the order of elements of the
            multiplicative group of integers modulo n,
        max_attempts: number of random x's to try, also an upper limit
            on the number of order_finder invocations.

    Returns:
        Non-trivial factor of n or None if no such factor was found.
        Factor k of n is trivial if it is 1 or n.
    """
    if sympy.isprime(n):
        return None
    if n % 2 == 0:
        return 2
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c
    for _ in range(max_attempts):
        x = random.randint(2, n - 1)
        c = math.gcd(x, n)
        if 1 < c < n:
            return c  # coverage: ignore
        r = order_finder(x, n)
        if r is None:
            continue  # coverage: ignore
        if r % 2 != 0:
            continue  # coverage: ignore
        y = x**(r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c
    return None  # coverage: ignore


def main(
        n: int,
        order_finder: Callable[[int, int], Optional[int]] = naive_order_finder,
):
    if n < 2:
        raise ValueError(
            f'Invalid input {n}, expected positive integer greater than one.')

    d = find_factor(n, order_finder)

    if d is None:
        print(f'No non-trivial factor of {n} found. It is probably a prime.')
    else:
        print(f'{d} is a non-trivial factor of {n}')

        assert 1 < d < n
        assert n % d == 0


if __name__ == '__main__':
    # coverage: ignore
    ORDER_FINDERS = {
        'naive': naive_order_finder,
        'quantum': quantum_order_finder,
    }
    args = parser.parse_args()
    main(n=args.n, order_finder=ORDER_FINDERS[args.order_finder])
