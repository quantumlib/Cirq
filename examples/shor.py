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
import math
import random

from typing import Callable, Optional

import sympy

parser = argparse.ArgumentParser(description='Factorization demo.')
parser.add_argument('n', type=int, help='composite integer to factor')


def naive_order_finder(x: int, n: int) -> int:
    """Computes smallest positive r such that x^r mod n == 1.

    Args:
        x: integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: modulus of the multiplicative group.

    Returns:
        Smallest positive integer r such that x^r == 1 mod n.

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


def quantum_order_finder(x: int, n: int) -> int:
    """Computes smallest positive r such that x^r mod n == 1."""
    raise NotImplementedError('Quantum order finder is not implemented yet.')


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
                order_finder: Callable[[int, int], int],
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
        if r % 2 != 0:
            continue  # coverage: ignore
        y = x**(r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c
    return None  # coverage: ignore


def main(n: Optional[int] = None):
    if n is None:
        n = parser.parse_args().n  # coverage: ignore
    if n < 2:
        raise ValueError(
            f'Invalid input {n}, expected positive integer greater than one.')

    d = find_factor(n, order_finder=naive_order_finder)

    if d is None:
        print(f'No non-trivial factor of {n} found. It is probably a prime.')
    else:
        print(f'{d} is a non-trivial factor of {n}')

        assert 1 < d < n
        assert n % d == 0


if __name__ == '__main__':
    main()
