# Copyright 2022 The Cirq Developers
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

"""Utilities to compute readout confusion matrix and use it for readout error mitigation."""

import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class TensoredConfusionMatrices:
    """Store and use confusion matrices for readout error mitigation on sets of qubits.

    The confusion matrix (CM) for one qubit is:

        [ Pr(0|0) Pr(1|0) ]
        [ Pr(1|0) Pr(1|1) ]

    where Pr(i | j) = Probability of observing state "i" given state "j" was prepared.

    Similarly, the confusion matrix for two qubits is:

        ⎡ Pr(00|00) Pr(01|00) Pr(10|00) Pr(11|00) ⎤
        ⎢ Pr(00|01) Pr(01|01) Pr(10|01) Pr(11|01) ⎥
        ⎢ Pr(00|10) Pr(01|10) Pr(10|10) Pr(11|10) ⎥
        ⎣ Pr(00|11) Pr(01|11) Pr(10|11) Pr(11|11) ⎦

    where Pr(ij | pq) = Probability of observing “ij” given state “pq” was prepared.

    This class can be used to
     - Store a list of confusion matrices computed for a list of qubit patterns.
     - Build a single confusion / correction matrix for entire set of calibrated qubits using the
       smaller individual confusion matrices for specific qubit patterns.
     - Apply readout corrections to observed frequencies / output probabilities.

    Use `cirq.measure_confusion_matrix(sampler, qubits, repetitions)` to perform
    an experiment on `sampler` and construct the `cirq.TensoredConfusionMatrices` object.
    """

    def __init__(
        self,
        confusion_matrices: Union[np.ndarray, Sequence[np.ndarray]],
        measure_qubits: Union[Sequence['cirq.Qid'], Sequence[Sequence['cirq.Qid']]],
        *,
        repetitions: int,
        timestamp: float,
    ):
        """Initializes `cirq.TensoredConfusionMatrices`.

        `confusion_matrices[i]` should correspond to the qubit sequence `measure_qubits[i]`.

        Args:
            confusion_matrices: Sequence of confusion matrices, computed for qubit patterns present
                                in `measure_qubits`. A single confusion matrix is also accepted.
            measure_qubits: Sequence of smaller qubit patterns, for which the confusion matrices
                            were computed. A single qubit pattern is also accepted. Note that the
                            each qubit pattern is a sequence of qubits used to label the axes of
                            the corresponding confusion matrix.
            repetitions:    The number of repetitions that were used to estimate the confusion
                            matrices.
            timestamp:      The time the data was taken, in seconds since the epoch.

        Raises:
            ValueError: If length of `confusion_matrices` and `measure_qubits` is different or if
                        the shape of any confusion matrix does not match the corresponding qubit
                        pattern.
        """
        if len(measure_qubits) == 0:
            raise ValueError(f"measure_qubits cannot be empty.")
        if isinstance(confusion_matrices, np.ndarray):
            confusion_matrices = [confusion_matrices]
        measure_qubits = cast(
            Sequence[Sequence['cirq.Qid']],
            [measure_qubits] if isinstance(measure_qubits[0], ops.Qid) else measure_qubits,
        )
        if len(confusion_matrices) != len(measure_qubits):
            raise ValueError(
                f"len(confusion_matrices): {len(confusion_matrices)} should be equal to "
                f"len(measure_qubits): {len(measure_qubits)}"
            )
        for i, (cm, q) in enumerate(zip(confusion_matrices, measure_qubits)):
            if cm.shape != (2 ** len(q),) * 2:
                raise ValueError(
                    f"Shape mismatch for confusion matrix {cm} at index {i} corresponding to {q}."
                    f"Confusion Matrix shape {cm.shape} should match {(2 ** len(q),) * 2}"
                )

        self._timestamp = timestamp
        self._repetitions = repetitions
        self._confusion_matrices = tuple(confusion_matrices)
        self._measure_qubits = tuple(tuple(q) for q in measure_qubits)
        self._qubits = tuple(sorted(set(q for ql in measure_qubits for q in ql)))
        self._qubits_to_idx = {q: i for i, q in enumerate(self._qubits)}
        self._cache: Dict[Tuple['cirq.Qid', ...], np.ndarray] = {}
        if sum(len(q) for q in self._measure_qubits) != len(self._qubits):
            raise ValueError(f"Repeated qubits not allowed in measure_qubits: {measure_qubits}.")

    @property
    def repetitions(self) -> int:
        """The number of repetitions that were used to estimate the confusion matrices."""
        return self._repetitions

    @property
    def timestamp(self) -> float:
        """The time the data for confusion matrix estimation was taken, in seconds since epoch."""
        return self._timestamp

    @property
    def confusion_matrices(self) -> Tuple[np.ndarray, ...]:
        """List of confusion matrices corresponding to `measure_qubits` qubit pattern."""
        return self._confusion_matrices

    @property
    def measure_qubits(self) -> Tuple[Tuple['cirq.Qid', ...], ...]:
        """Calibrated qubit pattern for which individual confusion matrices were computed."""
        return self._measure_qubits

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Sorted list of all calibrated qubits."""
        return self._qubits

    def _get_vars(self, qubit_pattern: Sequence['cirq.Qid']) -> List[int]:
        in_vars = [2 * self._qubits_to_idx[q] for q in qubit_pattern]
        out_vars = [2 * self._qubits_to_idx[q] + 1 for q in qubit_pattern]
        return in_vars + out_vars

    def _confusion_matrix(self, qubits: Sequence['cirq.Qid']) -> np.ndarray:
        ein_input = []
        for qs, cm in zip(self.measure_qubits, self.confusion_matrices):
            ein_input += [cm.reshape((2, 2) * len(qs)), self._get_vars(qs)]
        ein_out = self._get_vars(qubits)
        ret = np.einsum(*ein_input, ein_out).reshape((2 ** len(qubits),) * 2)
        return ret / ret.sum(axis=1)

    def confusion_matrix(self, qubits: Optional[Sequence['cirq.Qid']] = None) -> np.ndarray:
        """Returns a single confusion matrix constructed for the given set of qubits.

        The single `2 ** len(qubits) x 2 ** len(qubits)` confusion matrix is constructed
        using the individual smaller `self.confusion_matrices` by applying necessary
        matrix transpose / kron / partial trace operations.

        Args:
            qubits: The qubits representing the subspace for which a confusion matrix should be
                    constructed. By default, uses all qubits in sorted order, i.e. `self.qubits`.
                    Note that ordering of qubits sets the basis ordering of the returned matrix.

        Returns:
            Confusion matrix for subspace corresponding to `qubits`.

        Raises:
            ValueError: If `qubits` is not a subset of `self.qubits`.
        """

        if qubits is None:
            qubits = self.qubits
        if any(q not in self.qubits for q in qubits):
            raise ValueError(f"qubits {qubits} should be a subset of self.qubits {self.qubits}.")
        key = tuple(qubits)
        if key not in self._cache:
            self._cache[key] = self._confusion_matrix(qubits)
        return self._cache[key]

    def correction_matrix(self, qubits: Optional[Sequence['cirq.Qid']] = None) -> np.ndarray:
        """Returns a single correction matrix constructed for the given set of qubits.

        A correction matrix is the inverse of confusion matrix and can be used to apply corrections
        to observed frequencies / probabilities to compensate for the readout error.
        A Moore–Penrose Pseudo inverse of the confusion matrix is computed to get the correction
        matrix.

        Args:
            qubits: The qubits representing the subspace for which a correction matrix should be
                    constructed. By default, uses all qubits in sorted order, i.e. `self.qubits`.
                    Note that ordering of qubits sets the basis ordering of the returned matrix.

        Returns:
            Correction matrix for subspace corresponding to `qubits`.

        Raises:
            ValueError: If `qubits` is not a subset of `self.qubits`.
        """

        if qubits is None:
            qubits = self.qubits
        if any(q not in self.qubits for q in qubits):
            raise ValueError(f"qubits {qubits} should be a subset of self.qubits {self.qubits}.")
        return np.linalg.pinv(self.confusion_matrix(qubits))

    def apply(
        self,
        result: np.ndarray,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        *,
        method='least_squares',
    ) -> np.ndarray:
        """Applies corrections to the observed `result` to compensate for readout error on qubits.

        The compensation can applied by the following methods:
         1. 'pseudo_inverse': The result is multiplied by the correction matrix, which is pseudo
                              inverse of confusion matrix corresponding to the subspace defined by
                              `qubits`.
         2. 'least_squares': Solves a constrained minimization problem to find optimal `x` s.t.
                                a) x >= 0
                                b) sum(x) == sum(result) and
                                c) sum((result - x @ confusion_matrix) ** 2) is minimized.

        Args:
            result: `(2 ** len(qubits), )` shaped numpy array containing observed frequencies /
                    probabilities.
            qubits: Sequence of qubits used for sampling to get `result`. By default, uses all
                    qubits in sorted order, i.e. `self.qubits`. Note that ordering of qubits sets
                    the basis ordering for the `result` argument.
            method: Correction Method. Should be either 'pseudo_inverse' or 'least_squares'.
                    Equal to `least_squares` by default.

        Returns:
              `(2 ** len(qubits), )` shaped numpy array corresponding to `result` with corrections.

        Raises:
            ValueError: If `result.shape` != `(2 ** len(qubits),)`.
            ValueError: If `least_squares` constrained minimization problem does not converge.
        """
        if qubits is None:
            qubits = self.qubits
        if result.shape != (2 ** len(qubits),):
            raise ValueError(f"result.shape {result.shape} should be {(2 ** len(qubits),)}.")
        if method not in ['pseudo_inverse', 'least_squares']:
            raise ValueError(f"method: {method} should be 'pseudo_inverse' or 'least_squares'.")

        if method == 'pseudo_inverse':
            return result @ self.correction_matrix(qubits)  # coverage: ignore

        # Least squares minimization.
        cm = self.confusion_matrix(qubits)

        def func(x):
            return np.sum((result - x @ cm) ** 2)

        constraints = {'type': 'eq', 'fun': lambda x: sum(result) - sum(x)}
        bounds = tuple((0, sum(result)) for _ in result)
        res = scipy.optimize.minimize(
            func, result, method='SLSQP', constraints=constraints, bounds=bounds
        )
        if res.success is False:  # coverage: ignore
            raise ValueError(  # coverage: ignore
                f"SLSQP optimization for constrained minimization "  # coverage: ignore
                f"did not converge. Result:\n{res}"  # coverage: ignore
            )  # coverage: ignore
        return res.x

    def __repr__(self) -> str:
        return (
            f"cirq.TensoredConfusionMatrices("
            f"[{','.join([proper_repr(cm) for cm in self.confusion_matrices])}],"
            f"{self.measure_qubits},"
            f"repetitions={self.repetitions},"
            f"timestamp={self.timestamp}"
            f")"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'confusion_matrices': self.confusion_matrices,
            'measure_qubits': self.measure_qubits,
            'repetitions': self.repetitions,
            'timestamp': self.timestamp,
        }

    @classmethod
    def _from_json_dict_(
        cls, confusion_matrices, measure_qubits, repetitions, timestamp, **kwargs
    ) -> 'TensoredConfusionMatrices':
        return cls(
            [np.asarray(cm) for cm in confusion_matrices],
            measure_qubits,
            repetitions=repetitions,
            timestamp=timestamp,
        )

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.qubits == other.qubits
            and self.repetitions == other.repetitions
            and self.timestamp == other.timestamp
            and all(
                np.allclose(cm, ocm, atol=atol)
                for cm, ocm in zip(self.confusion_matrices, other.confusion_matrices)
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.qubits == other.qubits
            and self.repetitions == other.repetitions
            and self.timestamp == other.timestamp
            and all(
                np.array_equal(cm, ocm)
                for cm, ocm in zip(self.confusion_matrices, other.confusion_matrices)
            )
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other


def measure_confusion_matrix(
    sampler: 'cirq.Sampler',
    qubits: Union[Sequence['cirq.Qid'], Sequence[Sequence['cirq.Qid']]],
    repetitions: int = 1000,
) -> TensoredConfusionMatrices:
    """Prepares `TensoredConfusionMatrices` for the n qubits in the input.

    The confusion matrix (CM) for two qubits is the following matrix:

        ⎡ Pr(00|00) Pr(01|00) Pr(10|00) Pr(11|00) ⎤
        ⎢ Pr(00|01) Pr(01|01) Pr(10|01) Pr(11|01) ⎥
        ⎢ Pr(00|10) Pr(01|10) Pr(10|10) Pr(11|10) ⎥
        ⎣ Pr(00|11) Pr(01|11) Pr(10|11) Pr(11|11) ⎦

    where Pr(ij | pq) = Probability of observing “ij” given state “pq” was prepared.

    Args:
        sampler: Sampler to collect the data from.
        qubits: Qubits for which the confusion matrix should be measured.
        repetitions: Number of times to sample each circuit for a confusion matrix row.
    """
    qubits = cast(
        Sequence[Sequence['cirq.Qid']], [qubits] if isinstance(qubits[0], ops.Qid) else qubits
    )
    confusion_matrices = []
    for qs in qubits:
        flip_symbols = sympy.symbols(f'flip_0:{len(qs)}')
        flip_circuit = circuits.Circuit(
            [ops.X(q) ** s for q, s in zip(qs, flip_symbols)], ops.measure(*qs)
        )
        sweeps = study.Product(*[study.Points(f'flip_{i}', [0, 1]) for i in range(len(qs))])
        results = sampler.run_sweep(flip_circuit, sweeps, repetitions=repetitions)
        confusion_matrices.append(
            np.asarray([vis.get_state_histogram(r) for r in results], dtype=float) / repetitions
        )
    return TensoredConfusionMatrices(
        confusion_matrices, qubits, repetitions=repetitions, timestamp=time.time()
    )
