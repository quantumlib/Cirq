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

r"""Numerically-optimized compilation of arbitrary two-qubit unitaries.

Implements the NuOp compilation technique of
[arXiv:2106.15490](https://arxiv.org/abs/2106.15490){:.external}
("Designing Calibration and Expressivity-Efficient Instruction Sets for
Quantum Computing", Murali et al.). An arbitrary two-qubit unitary $U_t$ is
decomposed into a template of `i` layers of a fixed hardware two-qubit "base"
gate sandwiched between parameterized single-qubit U3 gates:

    $$k_i \cdot G \cdot k_{i-1} \cdot \ldots \cdot G \cdot k_0 \approx U_t$$

where $G$ is the base gate and each $k_j = U3 \otimes U3$ is a pair of
single-qubit gates whose rotation angles ($6 (i + 1)$ real parameters in
total) are treated as continuous optimization variables. A BFGS numerical
optimizer maximizes the decomposition fidelity

    $$F_d = |\mathrm{Tr}(U_d^\dagger U_t)| / \dim(U_t)$$

between the target unitary $U_t$ and the unitary $U_d$ realized by the
template. The number of layers is grown from 1 until a target fidelity is
met. Optionally, given hardware error rates for the base gate(s), the
compiler instead maximizes the overall fidelity

    $$F_u = F_d \cdot F_h, \qquad F_h = \prod_g (1 - p_g)$$

where the product in $F_h$ runs over all gates $g$ in the decomposition and
$p_g$ is the error rate of gate $g$. This enables noise-adaptive approximate
compilation: fewer, noisier-but-more-reliable base gates are preferred when
hardware errors dominate, and the best base gate type is selected per target
unitary.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Sequence
from typing import Any, NamedTuple, TYPE_CHECKING

import numpy as np
import scipy.optimize

from cirq import value
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq

_SingleQubitGatePair = tuple[np.ndarray, np.ndarray]


class TwoQubitNumericalCompilationResult(NamedTuple):
    r"""Represents a numerical compilation of a target 2-qubit gate onto a base gate.

    This object encodes the relationship between 4x4 unitary operators

        $$U_{target} \approx k_i \cdot U_{base} \cdot k_{i-1} \cdot \ldots
            \cdot k_1 \cdot U_{base} \cdot k_0$$

    where $U_{target}, U_{base}$ are 2-local and $k_j$ are 1-local. Equality
    holds up to a global phase within `decomposition_fidelity`.

    Attributes:
        base_gate_unitary: 4x4 unitary denoting $U_{base}$ above. If several
            base gates were given to the compiler, this is the one that was
            selected.
        target_gate: 4x4 unitary denoting $U_{target}$ above.
        local_unitaries: Sequence of 2-tuples
            $(k_{00}, k_{01}), (k_{10}, k_{11}) \ldots$ where
            $k_j = k_{j0} \otimes k_{j1}$ in the product above.
            Each $k_{j0}, k_{j1}$ is a 2x2 unitary. Has `num_base_gates + 1`
            entries.
        actual_gate: 4x4 unitary denoting the right hand side above, ideally
            equal to $U_{target}$ up to a global phase.
        decomposition_fidelity: $F_d = |\mathrm{Tr}(U_d^\dagger U_t)| / 4$,
            the Hilbert-Schmidt overlap between the realized and target
            unitaries. 1.0 means an exact decomposition up to global phase.
        hardware_fidelity: $F_h = \prod_g (1 - p_g)$ over all gates $g$ in
            the decomposition, or None if no error rates were provided.
        num_base_gates: Number of base gate applications (layers) used.
        success: Whether `decomposition_fidelity` meets the target fidelity
            requested from the compiler.
    """

    base_gate_unitary: np.ndarray
    target_gate: np.ndarray
    local_unitaries: tuple[_SingleQubitGatePair, ...]
    actual_gate: np.ndarray
    decomposition_fidelity: float
    hardware_fidelity: float | None
    num_base_gates: int
    success: bool


def _u3_matrix(alpha: float, beta: float, lam: float) -> np.ndarray:
    r"""Single-qubit U3 gate used in the template.

        $$U3(\alpha, \beta, \lambda) = \begin{pmatrix}
            \cos(\alpha/2) & -e^{i\lambda} \sin(\alpha/2) \\
            e^{i\beta} \sin(\alpha/2) & e^{i(\beta+\lambda)} \cos(\alpha/2)
        \end{pmatrix}$$

    which parametrizes all of SU(2) up to a global phase.
    """
    cos_a, sin_a = np.cos(alpha / 2), np.sin(alpha / 2)
    return np.array(
        [
            [cos_a, -np.exp(1j * lam) * sin_a],
            [np.exp(1j * beta) * sin_a, np.exp(1j * (beta + lam)) * cos_a],
        ]
    )


def _template_unitary(params: np.ndarray, base_gate: np.ndarray, num_layers: int) -> np.ndarray:
    r"""Unitary realized by the layered template for the given angles.

    Args:
        params: Array of shape (num_layers + 1, 2, 3) with the U3 angles of
            the single-qubit gate pairs $k_0, \ldots, k_i$.
        base_gate: 4x4 unitary of the base two-qubit gate.
        num_layers: Number of base gate applications.
    """
    blocks = params.reshape(num_layers + 1, 2, 3)
    ud = np.eye(4, dtype=complex)
    for block in range(num_layers + 1):
        k = np.kron(_u3_matrix(*blocks[block, 0]), _u3_matrix(*blocks[block, 1]))
        ud = k @ ud
        if block < num_layers:
            ud = base_gate @ ud
    return ud


def _decomposition_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    r"""$F_d = |\mathrm{Tr}(U_d^\dagger U_t)| / \dim(U_t)$."""
    return abs(np.trace(actual.conj().T @ target)) / target.shape[0]


def _optimize_template(
    target: np.ndarray,
    base_gate: np.ndarray,
    num_layers: int,
    rng: np.random.RandomState,
    num_restarts: int,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    r"""Best template angles for a fixed base gate and number of layers.

    Runs BFGS from `num_restarts` random starting points and returns the angles
    achieving the highest decomposition fidelity, along with that fidelity.

    The optimized objective is the infidelity $1 - F_d$ with
    $F_d = |\mathrm{Tr}(U_d^\dagger U_t)| / 4$.
    """
    num_params = 6 * (num_layers + 1)

    def objective(x: np.ndarray) -> float:
        ud = _template_unitary(x, base_gate, num_layers)
        return 1.0 - _decomposition_fidelity(ud, target)

    best_x: np.ndarray | None = None
    best_fidelity = -np.inf
    bounds = [(0.0, 2 * np.pi)] * num_params
    for _ in range(num_restarts):
        x0 = rng.uniform(0.0, 2 * np.pi, size=num_params)
        res = scipy.optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-14},
        )
        fidelity = 1.0 - res.fun if np.isfinite(res.fun) else -np.inf
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_x = res.x
    if best_x is None:
        raise RuntimeError(
            'All optimization runs produced a non-finite objective value; '
            'check that target and base_gate are valid (finite) unitaries.'
        )
    return best_x, best_fidelity


def _result_from_angles(
    params: np.ndarray,
    base_gate: np.ndarray,
    target: np.ndarray,
    num_layers: int,
    hardware_fidelity: float | None,
    success: bool,
) -> TwoQubitNumericalCompilationResult:
    actual = _template_unitary(params, base_gate, num_layers)
    blocks = params.reshape(num_layers + 1, 2, 3)
    local_unitaries = tuple(
        (_u3_matrix(*blocks[b, 0]), _u3_matrix(*blocks[b, 1])) for b in range(num_layers + 1)
    )
    return TwoQubitNumericalCompilationResult(
        base_gate_unitary=base_gate,
        target_gate=target,
        local_unitaries=local_unitaries,
        actual_gate=actual,
        decomposition_fidelity=_decomposition_fidelity(actual, target),
        hardware_fidelity=hardware_fidelity,
        num_base_gates=num_layers,
        success=success,
    )


def two_qubit_gate_numerical_compilation(
    target_unitary: np.ndarray,
    base_gates: np.ndarray | Sequence[np.ndarray],
    *,
    target_fidelity: float = 1 - 1e-8,
    max_layers: int = 3,
    base_gate_error_rates: Sequence[float] | None = None,
    single_qubit_error_rate: float = 0.0,
    num_restarts: int = 3,
    maxiter: int = 1000,
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> TwoQubitNumericalCompilationResult:
    r"""Compile a two-qubit unitary onto hardware base gate(s) via numerical optimization.

    Implements the NuOp compilation technique of
    [arXiv:2106.15490](https://arxiv.org/abs/2106.15490){:.external}.
    The target unitary is decomposed into `i` layers of a base two-qubit gate
    sandwiched between parameterized single-qubit gates,

    $k_i \cdot G \cdot k_{i-1} \cdot \ldots \cdot G \cdot k_0 \approx U_t$,

    where the angles of the 1-local unitaries $k_j$ are found by a BFGS
    numerical optimizer maximizing the decomposition fidelity
    $F_d = |\mathrm{Tr}(U_d^\dagger U_t)| / 4$.

    The number of layers is grown from 1 upward, optimizing each template in
    turn, and the first decomposition whose fidelity meets `target_fidelity`
    is returned. With several base gates, all of them are tried at each layer
    count, so the returned decomposition is the one with the fewest layers
    across all gate types. If no template with at most `max_layers` layers
    meets `target_fidelity`, the highest-fidelity decomposition found is
    returned with `success` set to False.

    If `base_gate_error_rates` is given, the compiler additionally accounts
    for hardware noise: instead of stopping at the first decomposition meeting
    `target_fidelity`, all (base gate, layer count) combinations up to
    `max_layers` are optimized, and the decomposition maximizing the overall
    fidelity $F_u = F_d \cdot F_h$ is returned, where the hardware fidelity
    $F_h$ is the product of (1 - error_rate) over all gates in the
    decomposition. This can deliberately select an approximate decomposition
    with fewer base gates when hardware noise dominates the decomposition
    error, and selects the best base gate type for each target unitary.

    Args:
        target_unitary: The 4x4 unitary to compile.
        base_gates: A single 4x4 base gate unitary, or a sequence of them.
            Each base gate must be entangling for arbitrary targets to be
            compilable.
        target_fidelity: Decomposition fidelity threshold. Layer growth stops
            once it is met; `success` in the result reports whether the
            returned decomposition meets it. Should be a float slightly below
            1, e.g. 1 - 1e-8 for exact compilation or 0.99 for approximate.
        max_layers: Maximum number of base gate applications allowed.
        base_gate_error_rates: Optional hardware error rate of each base gate,
            one per entry of `base_gates`. When given, the compiler maximizes
            the overall fidelity $F_u = F_d \cdot F_h$ over all (base gate,
            layer count) combinations instead of stopping at the first
            decomposition meeting `target_fidelity`.
        single_qubit_error_rate: Optional hardware error rate of each
            single-qubit gate, folded into the hardware fidelity $F_h$ when
            `base_gate_error_rates` is given.
        num_restarts: Number of random restarts of the BFGS optimizer per
            (base gate, layer count) pair. Higher values improve accuracy at
            the cost of runtime.
        maxiter: Maximum number of iterations of each BFGS run.
        random_state: Random state or seed used to generate the optimizer's
            starting points.

    Returns:
        A TwoQubitNumericalCompilationResult with the best decomposition found.

    Raises:
        ValueError: If `target_unitary` is not 4x4, `base_gates` is empty or
            malformed, `base_gate_error_rates` does not match `base_gates`,
            or `target_fidelity`, `max_layers`, `num_restarts`, `maxiter`,
            `base_gate_error_rates` or `single_qubit_error_rate` are out of
            range.
        RuntimeError: If every optimization run produced a non-finite
            objective value, e.g. because of non-finite inputs.
    """
    target = np.asarray(target_unitary)
    if target.shape != (4, 4):
        raise ValueError(f'target_unitary must have shape (4, 4), got {target.shape}')
    if not 0 < target_fidelity < 1:
        raise ValueError(f'target_fidelity must be in (0, 1), got {target_fidelity}')
    if max_layers < 1:
        raise ValueError(f'max_layers must be at least 1, got {max_layers}')
    if num_restarts < 1:
        raise ValueError(f'num_restarts must be at least 1, got {num_restarts}')
    if maxiter < 1:
        raise ValueError(f'maxiter must be at least 1, got {maxiter}')

    gates = np.asarray(base_gates, dtype=complex)
    if gates.shape == (4, 4):
        gates = gates[np.newaxis]
    if gates.ndim != 3 or gates.shape[1:] != (4, 4) or gates.shape[0] == 0:
        raise ValueError(
            f'base_gates must be a 4x4 unitary or a sequence of them, got {gates.shape}'
        )
    if base_gate_error_rates is not None:
        if len(base_gate_error_rates) != gates.shape[0]:
            raise ValueError(
                f'Expected one error rate per base gate ({gates.shape[0]}), '
                f'got {len(base_gate_error_rates)}'
            )
        if any(not 0 <= p <= 1 for p in base_gate_error_rates):
            raise ValueError(
                f'base_gate_error_rates must be in [0, 1], got {base_gate_error_rates}'
            )
    if not 0 <= single_qubit_error_rate <= 1:
        raise ValueError(
            f'single_qubit_error_rate must be in [0, 1], got {single_qubit_error_rate}'
        )

    rng = value.parse_random_state(random_state)

    best: TwoQubitNumericalCompilationResult | None = None
    best_overall = -np.inf
    if base_gate_error_rates is None:
        # Layer growth: iterate layer counts in increasing order and return the
        # first (base gate, layer count) meeting the target fidelity. Fall back
        # to the highest-fidelity decomposition if none meets the threshold.
        for num_layers in range(1, max_layers + 1):
            for base_gate in gates:
                params, fidelity = _optimize_template(
                    target, base_gate, num_layers, rng, num_restarts, maxiter
                )
                result = _result_from_angles(
                    params, base_gate, target, num_layers, None, fidelity >= target_fidelity
                )
                if fidelity > best_overall:
                    best_overall = fidelity
                    best = result
                if fidelity >= target_fidelity:
                    return result
    else:
        # With error rates given, maximize the overall fidelity Fu = Fd * Fh
        # over all (base gate, layer count) combinations.
        for gate_ind, base_gate in enumerate(gates):
            for num_layers in range(1, max_layers + 1):
                params, fidelity = _optimize_template(
                    target, base_gate, num_layers, rng, num_restarts, maxiter
                )
                hardware_fidelity = (1 - base_gate_error_rates[gate_ind]) ** num_layers
                hardware_fidelity *= (1 - single_qubit_error_rate) ** (2 * (num_layers + 1))
                if fidelity * hardware_fidelity > best_overall:
                    best_overall = fidelity * hardware_fidelity
                    best = _result_from_angles(
                        params,
                        base_gate,
                        target,
                        num_layers,
                        hardware_fidelity,
                        success=fidelity >= target_fidelity,
                    )

    assert best is not None
    return best


@dataclasses.dataclass(frozen=True)
class TwoQubitNumericalCompiler:
    r"""A two-qubit gate compiler based on numerical optimization (NuOp).

    Holds a fixed set of hardware base gates (and optionally their error
    rates) so that arbitrary two-qubit unitaries can be compiled repeatedly
    with the same configuration via `compile_two_qubit_gate`. See
    `cirq.two_qubit_gate_numerical_compilation` for details of the algorithm.

    This class supports JSON serialization, but only when `random_state` is
    None or an integer seed; a live numpy random generator cannot be
    serialized.
    """

    base_gates: tuple[np.ndarray, ...]
    base_gate_error_rates: tuple[float, ...] | None = None
    target_fidelity: float = 1 - 1e-8
    max_layers: int = 3
    single_qubit_error_rate: float = 0.0
    num_restarts: int = 3
    maxiter: int = 1000
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None

    def compile_two_qubit_gate(self, unitary: np.ndarray) -> TwoQubitNumericalCompilationResult:
        """Compile the given 4x4 unitary onto this compiler's base gate(s)."""
        return two_qubit_gate_numerical_compilation(
            unitary,
            self.base_gates,
            target_fidelity=self.target_fidelity,
            max_layers=self.max_layers,
            base_gate_error_rates=self.base_gate_error_rates,
            single_qubit_error_rate=self.single_qubit_error_rate,
            num_restarts=self.num_restarts,
            maxiter=self.maxiter,
            random_state=self.random_state,
        )

    def _json_dict_(self) -> dict[str, Any]:
        if self.random_state is not None and not isinstance(self.random_state, numbers.Integral):
            raise ValueError(
                'TwoQubitNumericalCompiler can only be JSON-serialized when '
                'random_state is None or an integer seed, got '
                f'{type(self.random_state).__name__}.'
            )
        return {
            'base_gates': [gate.tolist() for gate in self.base_gates],
            'base_gate_error_rates': self.base_gate_error_rates,
            'target_fidelity': self.target_fidelity,
            'max_layers': self.max_layers,
            'single_qubit_error_rate': self.single_qubit_error_rate,
            'num_restarts': self.num_restarts,
            'maxiter': self.maxiter,
            'random_state': self.random_state,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        base_gates,
        base_gate_error_rates,
        target_fidelity,
        max_layers,
        single_qubit_error_rate,
        num_restarts,
        maxiter,
        random_state,
        **kwargs,
    ):
        return cls(
            base_gates=tuple(np.asarray(gate, dtype=complex) for gate in base_gates),
            base_gate_error_rates=(
                None if base_gate_error_rates is None else tuple(base_gate_error_rates)
            ),
            target_fidelity=target_fidelity,
            max_layers=max_layers,
            single_qubit_error_rate=single_qubit_error_rate,
            num_restarts=num_restarts,
            maxiter=maxiter,
            random_state=random_state,
        )

    def __repr__(self) -> str:
        gates = ', '.join(proper_repr(gate) for gate in self.base_gates)
        if len(self.base_gates) == 1:
            gates += ','
        return (
            f'cirq.TwoQubitNumericalCompiler(base_gates=({gates}), '
            f'base_gate_error_rates={proper_repr(self.base_gate_error_rates)}, '
            f'target_fidelity={self.target_fidelity!r}, '
            f'max_layers={self.max_layers!r}, '
            f'single_qubit_error_rate={self.single_qubit_error_rate!r}, '
            f'num_restarts={self.num_restarts!r}, '
            f'maxiter={self.maxiter!r}, '
            f'random_state={self.random_state!r})'
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            len(self.base_gates) == len(other.base_gates)
            and all(
                np.array_equal(own, theirs)
                for own, theirs in zip(self.base_gates, other.base_gates)
            )
            and self.base_gate_error_rates == other.base_gate_error_rates
            and self.target_fidelity == other.target_fidelity
            and self.max_layers == other.max_layers
            and self.single_qubit_error_rate == other.single_qubit_error_rate
            and self.num_restarts == other.num_restarts
            and self.maxiter == other.maxiter
            and self.random_state == other.random_state
        )
