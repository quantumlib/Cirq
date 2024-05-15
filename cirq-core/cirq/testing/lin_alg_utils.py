# Copyright 2018 The Cirq Developers
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
"""A testing class with utilities for checking linear algebra."""

from typing import Optional, TYPE_CHECKING

import numpy as np

from cirq import linalg, value

if TYPE_CHECKING:
    import cirq


def random_superposition(
    dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
) -> np.ndarray:
    """Returns a random unit-length vector from the uniform distribution.

    Args:
        dim: The dimension of the vector.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled unit-length vector.
    """
    random_state = value.parse_random_state(random_state)

    state_vector = random_state.randn(dim).astype(complex)
    state_vector += 1j * random_state.randn(dim)
    state_vector /= np.linalg.norm(state_vector)
    return state_vector


def random_density_matrix(
    dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
) -> np.ndarray:
    """Returns a random density matrix distributed with Hilbert-Schmidt measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed to use for random number generation.

    Returns:
        The sampled density matrix.

    Reference:
        'Random Bures mixed states and the distribution of their purity'
        https://arxiv.org/abs/0909.5094
    """
    random_state = value.parse_random_state(random_state)

    mat = random_state.randn(dim, dim) + 1j * random_state.randn(dim, dim)
    mat = mat @ mat.T.conj()
    return mat / np.trace(mat)


def random_unitary(
    dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
) -> np.ndarray:
    """Returns a random unitary matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed to use for random number generation.

    Returns:
        The sampled unitary matrix.

    References:
        'How to generate random matrices from the classical compact groups'
        http://arxiv.org/abs/math-ph/0609050
    """
    random_state = value.parse_random_state(random_state)

    z = random_state.randn(dim, dim) + 1j * random_state.randn(dim, dim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / abs(d))


def random_orthogonal(
    dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
) -> np.ndarray:
    """Returns a random orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled orthogonal matrix.

    References:
        'How to generate random matrices from the classical compact groups'
        http://arxiv.org/abs/math-ph/0609050
    """
    random_state = value.parse_random_state(random_state)

    m = random_state.randn(dim, dim)
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    return q * (d / abs(d))


def random_special_unitary(
    dim: int, *, random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Returns a random special unitary distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled special unitary.
    """
    r = random_unitary(dim, random_state=random_state)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[0, :] /= np.linalg.det(r)
    return r


def random_special_orthogonal(
    dim: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
) -> np.ndarray:
    """Returns a random special orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        random_state: A seed (int) or `np.random.RandomState` class to use when
            generating random values. If not set, defaults to using the module
            methods in `np.random`.

    Returns:
        The sampled special orthogonal matrix.
    """
    m = random_orthogonal(dim, random_state=random_state)
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.linalg.det(m) < 0:
            m[0, :] *= -1
    return m


def assert_allclose_up_to_global_phase(
    actual: np.ndarray,
    desired: np.ndarray,
    *,  # Forces keyword args.
    rtol: float = 1e-7,
    atol: float,  # Require atol to be specified
    equal_nan: bool = True,
    err_msg: str = '',
    verbose: bool = True,
) -> None:
    """Checks if a ~= b * exp(i t) for some t.

    Args:
        actual: A numpy array.
        desired: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error
            message.

    Raises:
        AssertionError: The matrices aren't nearly equal up to global phase.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    actual, desired = linalg.match_global_phase(actual, desired)
    np.testing.assert_allclose(
        actual=actual,
        desired=desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )
