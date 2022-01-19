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

import functools
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast

import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class ReadoutConfusionMatrix:
    """Store and use confusion matrices for readout error mitigation on sets of qubits.

    The confusion matrix (CM) for two qubits is the following matrix:

        ⎡ Pr(00o|00a) Pr(01o|00a) Pr(10o|00a) Pr(11o|00a) ⎤
        ⎢ Pr(00o|01a) Pr(01o|01a) Pr(10o|01a) Pr(11o|01a) ⎥
        ⎢ Pr(00o|10a) Pr(01o|10a) Pr(10o|10a) Pr(11o|10a) ⎥
        ⎣ Pr(00o|11a) Pr(01o|11a) Pr(10o|11a) Pr(11o|11a) ⎦

    where Pr(ij | pq) = Probability of observing “ij” given state “pq” was prepared.

    This class can be used to
     - Store a list of confusion matrices computed for a list of qubit patterns.
     - Build a single confusion / correction matrix for entire set of calibrated qubits using the
       smaller individual confusion matrices for specific qubit patterns.
     - Apply readout corrections to observed frequencies / output probabilities.

    Use `cirq.measure_confusion_matrix(sampler, qubits, repetitions)` to perform
    an experiment on `sampler` and construct the `cirq.ReadoutConfusionMatrix` object.
    """

    def __init__(
        self,
        confusion_matrices: Union[np.ndarray, Sequence[np.ndarray]],
        measure_qubits: Union[Sequence['cirq.Qid'], Sequence[Sequence['cirq.Qid']]],
    ):
        """Initializes `cirq.ReadoutConfusionMatrix`.

        `confusion_matrices[i]` should correspond to the qubit sequence `measure_qubits[i]`.

        Args:
            confusion_matrices: Sequence of Confusion matrices, computed for qubit patterns present
                                in `measure_qubits`. A single confusion matrix is also accepted.
            measure_qubits: Sequence of smaller qubit patterns, for which the confusion matrices
                            were computed. A single qubit pattern is also accepted.
        Raises:
            ValueError: If length of `confusion_matrices` and `measure_qubits` is different or if
                        the shape of any confusion matrix does not match the corresponding qubit
                        pattern.
        """
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
        self._confusion_matrices = list(confusion_matrices)
        self._measure_qubits = [list(q) for q in measure_qubits]
        self._qubits = sorted(set(q for ql in measure_qubits for q in ql))

    @property
    def confusion_matrices(self) -> List[np.ndarray]:
        """List of confusion matrices corresponding to `measure_qubits` qubit pattern."""
        return self._confusion_matrices

    @property
    def measure_qubits(self) -> List[List['cirq.Qid']]:
        """Calibrated qubit pattern for which individual confusion matrices were computed."""
        return self._measure_qubits

    @property
    def qubits(self) -> List['cirq.Qid']:
        """Sorted list of all calibrated qubits."""
        return self._qubits

    def _get_vars(self, qubit_pattern: Optional[Sequence[Sequence['cirq.Qid']]] = None):
        if qubit_pattern is None:
            qubit_pattern = self.measure_qubits
        abcd = "abcdefghijklmnopqrstuvwxyz"

        def qubits_to_abcd(qs: Sequence['cirq.Qid']):
            assert len(qs) <= len(abcd), "No. of qubits should be <= 26."
            ret = ''.join(abcd[self.qubits.index(q)] for q in qs)
            return ret + ret.upper()

        return ','.join(qubits_to_abcd(qs) for qs in qubit_pattern)

    @functools.lru_cache()
    def _confusion_matrix(self, qubits: Tuple['cirq.Qid']) -> np.ndarray:
        ret = np.einsum(
            f'{self._get_vars()}->{self._get_vars([qubits])}',
            *[
                cm.reshape((2, 2) * len(qs))
                for qs, cm in zip(self.measure_qubits, self.confusion_matrices)
            ],
        ).reshape((2 ** len(qubits),) * 2)
        return ret / ret.sum(axis=1)

    def confusion_matrix(self, qubits: Optional[Sequence['cirq.Qid']] = None) -> np.ndarray:
        """Returns a single confusion matrix constructed for the given set of qubits.

        The single `2 ** len(qubits) x 2 ** len(qubits)` confusion matrix is constructed
        using the individual smaller `self.confusion_matrices` by applying necessary
        matrix transpose / kron / partial trace operations.

        Args:
            qubits: The qubits representing the subspace for which a confusion matrix should be
                    constructed. By default, uses all qubits in sorted order, i.e. `self.qubits`.

        Returns:
            Confusion matrix for subspace corresponding to `qubits`.

        Raises:
            ValueError: If `qubits` is not a subset of `self.qubits`.
        """

        if qubits is None:
            qubits = self.qubits
        if any(q not in self.qubits for q in qubits):
            raise ValueError(f"qubits {qubits} should be a subset of self.qubits {self.qubits}.")
        return self._confusion_matrix(tuple(qubits))

    def correction_matrix(self, qubits: Optional[Sequence['cirq.Qid']] = None) -> np.ndarray:
        """Returns a single correction matrix constructed for the given set of qubits.

        A correction matrix is the inverse of confusion matrix and can be used to apply corrections
        to observed frequencies / probabilities to compensate for the readout error.
        A Moore–Penrose Pseudo inverse of the confusion matrix is computed to get the correction
        matrix.

        Args:
            qubits: The qubits representing the subspace for which a correction matrix should be
                    constructed. By default, uses all qubits in sorted order, i.e. `self.qubits`.

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

        The compensation is applied by multiplying the result with the correction matrix
        corresponding to the subspace defined by `qubits`.

        Args:
            result: `(2 ** len(qubits), )` shaped numpy array containing observed frequencies /
                    probabilities.
            qubits: Sequence of qubits used for sampling to get `result`. By default, uses all
                    qubits in sorted order, i.e. `self.qubits`.
            method: Correction Method. Should be either 'pseudo_inverse' or 'least_squares'.

        Returns:
              `(2 ** len(qubits), )` shaped numpy array corresponding to `result` with corrections.

        Raises:
            ValueError: if `result.shape` != `(2 ** len(qubits),)`.
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
            print(x.shape)
            return np.sum((result - x @ cm) ** 2)

        constraints = {'type': 'eq', 'fun': lambda x: sum(result) - sum(x)}
        bounds = tuple((0, sum(result)) for _ in result)
        res = scipy.optimize.minimize(
            func, result, method='SLSQP', constraints=constraints, bounds=bounds
        )
        return res.x

    def __repr__(self) -> str:
        return (
            f"cirq.ReadoutConfusionMatrix("
            f"[{','.join([proper_repr(cm) for cm in self.confusion_matrices])}],"
            f"{self.measure_qubits}"
            f")"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'confusion_matrices': self.confusion_matrices,
            'measure_qubits': self.measure_qubits,
        }

    @classmethod
    def _from_json_dict_(
        cls, confusion_matrices, measure_qubits, **kwargs
    ) -> 'ReadoutConfusionMatrix':
        return cls([np.asarray(cm) for cm in confusion_matrices], measure_qubits)

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.qubits == other.qubits and all(
            np.allclose(cm, ocm, atol=atol)
            for cm, ocm in zip(self.confusion_matrices, other.confusion_matrices)
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.qubits == other.qubits and all(
            np.array_equal(cm, ocm)
            for cm, ocm in zip(self.confusion_matrices, other.confusion_matrices)
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        vals = tuple(v for cm in self.confusion_matrices for _, v in np.ndenumerate(cm))
        return hash((ReadoutConfusionMatrix, vals, tuple(self.qubits)))


def measure_confusion_matrix(
    sampler: 'cirq.Sampler',
    qubits: Union[Sequence['cirq.Qid'], Sequence[Sequence['cirq.Qid']]],
    repetitions: int = 1000,
) -> ReadoutConfusionMatrix:
    """Prepares `ReadoutConfusionMatrix` for the n qubits in the input.

    The confusion matrix (CM) for two qubits is the following matrix:

        ⎡ Pr(00o|00a) Pr(01o|00a) Pr(10o|00a) Pr(11o|00a) ⎤
        ⎢ Pr(00o|01a) Pr(01o|01a) Pr(10o|01a) Pr(11o|01a) ⎥
        ⎢ Pr(00o|10a) Pr(01o|10a) Pr(10o|10a) Pr(11o|10a) ⎥
        ⎣ Pr(00o|11a) Pr(01o|11a) Pr(10o|11a) Pr(11o|11a) ⎦

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
        results = sampler.run_batch(
            [
                circuits.Circuit(
                    [ops.X(q) ** ((state >> i) & 1) for i, q in enumerate(qs[::-1])],
                    ops.measure(*qs),
                )
                for state in range(2 ** len(qs))
            ],
            repetitions=repetitions,
        )
        confusion_matrices.append(
            np.asarray([vis.get_state_histogram(r[0]) for r in results], dtype=float) / repetitions
        )
    return ReadoutConfusionMatrix(confusion_matrices, qubits)
