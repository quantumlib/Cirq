# Copyright 2019 The Cirq Developers
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

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, TYPE_CHECKING, Tuple
import dataclasses
import numpy as np
from matplotlib import pyplot as plt
from cirq import _compat, _import, protocols

if TYPE_CHECKING:
    import cirq

CrossEntropyPair = NamedTuple('CrossEntropyPair', [('num_cycle', int), ('xeb_fidelity', float)])
SpecklePurityPair = NamedTuple('SpecklePurityPair', [('num_cycle', int), ('purity', float)])

optimize = _import.LazyLoader("optimize", globals(), "scipy.optimize")


@dataclasses.dataclass
class CrossEntropyDepolarizingModel:
    """A depolarizing noise model for cross entropy benchmarking.

    The depolarizing channel maps a density matrix ρ as

        ρ → p_eff ρ + (1 - p_eff) I / D

    where I / D is the maximally mixed state and p_eff is between 0 and 1.
    It is used to model the effect of noise in certain quantum processes.
    This class models the noise that results from the execution of multiple
    layers, or cycles, of a random quantum circuit. In this model, p_eff for
    the whole process is separated into a part that is independent of the number
    of cycles (representing depolarization from state preparation and
    measurement errors), and a part that exhibits exponential decay with the
    number of cycles (representing depolarization from circuit execution
    errors). So p_eff is modeled as

        p_eff = S * p**d

    where d is the number of cycles, or depth, S is the part that is independent
    of depth, and p describes the exponential decay with depth. This class
    stores S and p, as well as possibly the covariance in their estimation from
    experimental data.

    Attributes:
        spam_depolarization: The depolarization constant for state preparation
            and measurement, i.e., S in p_eff = S * p**d.
        cycle_depolarization: The depolarization constant for circuit execution,
            i.e., p in p_eff = S * p**d.
        covariance: The estimated covariance in the estimation of
            `spam_depolarization` and `cycle_depolarization`, in that order.
    """

    spam_depolarization: float
    cycle_depolarization: float
    covariance: Optional[np.ndarray] = None


class SpecklePurityDepolarizingModel(CrossEntropyDepolarizingModel):
    """A depolarizing noise model for speckle purity benchmarking.

    In speckle purity benchmarking, the state ρ in the map

        ρ → p_eff ρ + (1 - p_eff) I / D

    is taken to be an unconstrained pure state. The purity of the resultant
    state is p_eff**2. The aim of speckle purity benchmarking is to measure the
    purity of the state resulting from applying a single XEB cycle to a pure
    state. This value is stored in the `purity` property of this class.
    """

    @property
    def purity(self) -> float:
        """The purity. Equal to p**2, where p is the cycle depolarization."""
        return self.cycle_depolarization**2


@_compat.deprecated_class(
    deadline='v0.16', fix=('Use cirq.experiments.xeb_fitting.XEBCharacterizationResult instead')
)
@dataclasses.dataclass(frozen=True)
class CrossEntropyResult:
    """Results from a cross-entropy benchmarking (XEB) experiment.

    May also include results from speckle purity benchmarking (SPB) performed
    concomitantly.

    Attributes:
        data: A sequence of NamedTuples, each of which contains two fields:
            num_cycle: the circuit depth as the number of cycles, where
            a cycle consists of a layer of single-qubit gates followed
            by a layer of two-qubit gates.
            xeb_fidelity: the XEB fidelity after the given cycle number.
        repetitions: The number of circuit repetitions used.
        purity_data: A sequence of NamedTuples, each of which contains two
            fields:
            num_cycle: the circuit depth as the number of cycles, where
            a cycle consists of a layer of single-qubit gates followed
            by a layer of two-qubit gates.
            purity: the purity after the given cycle number.
    """

    data: List[CrossEntropyPair]
    repetitions: int
    purity_data: Optional[List[SpecklePurityPair]] = None

    def plot(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the average XEB fidelity vs the number of cycles.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        num_cycles = [d.num_cycle for d in self.data]
        fidelities = [d.xeb_fidelity for d in self.data]
        ax.set_ylim([0, 1.1])
        ax.plot(num_cycles, fidelities, 'ro-', **plot_kwargs)
        ax.set_xlabel('Number of Cycles')
        ax.set_ylabel('XEB Fidelity')
        if show_plot:
            fig.show()
        return ax

    def depolarizing_model(self) -> CrossEntropyDepolarizingModel:
        """Fit a depolarizing error model for a cycle.

        Fits an exponential model f = S * p**d, where d is the number of cycles
        and f is the cross entropy fidelity for that number of cycles,
        using nonlinear least squares.

        Returns:
            A CrossEntropyDepolarizingModel object, which has attributes
            `spam_depolarization` representing the value S,
            `cycle_depolarization` representing the value p, and `covariance`
            representing the covariance in the estimation of S and p in that
            order.
        """
        depths, fidelities = zip(*self.data)
        params, covariance = _fit_exponential_decay(depths, fidelities)
        return CrossEntropyDepolarizingModel(
            spam_depolarization=params[0], cycle_depolarization=params[1], covariance=covariance
        )

    def purity_depolarizing_model(self) -> CrossEntropyDepolarizingModel:
        """Fit a depolarizing error model for a cycle to purity data.

        Fits an exponential model f = S * p**d, where d is the number of cycles
        and p**2 is the purity for that number of cycles, using nonlinear least
        squares.

        Returns:
            A SpecklePurityDepolarizingModel object, which has attributes
            `spam_depolarization` representing the value S,
            `cycle_depolarization` representing the value p, and `covariance`
            representing the covariance in the estimation of S and p in that
            order. It also has the property `purity` representing the purity
            p**2.

        Raises:
            ValueError: If no `purity_data` has been supplied to this class.
        """
        if self.purity_data is None:
            raise ValueError(
                'This CrossEntropyResult does not contain data '
                'from speckle purity benchmarking, so the '
                'purity depolarizing model cannot be computed.'
            )
        depths, purities = zip(*self.purity_data)
        params, covariance = _fit_exponential_decay(depths, np.sqrt(purities))
        return SpecklePurityDepolarizingModel(
            spam_depolarization=params[0], cycle_depolarization=params[1], covariance=covariance
        )

    @classmethod
    def _from_json_dict_(cls, data, repetitions, **kwargs):
        purity_data = kwargs.get('purity_data', None)
        if purity_data is not None:
            purity_data = [SpecklePurityPair(d, f) for d, f in purity_data]
        return cls(
            data=[CrossEntropyPair(d, f) for d, f in data],
            repetitions=repetitions,
            purity_data=purity_data,
        )

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)

    def __repr__(self) -> str:
        args = f'data={[tuple(p) for p in self.data]!r}, repetitions={self.repetitions!r}'
        if self.purity_data is not None:
            args += f', purity_data={[tuple(p) for p in self.purity_data]!r}'
        return f'cirq.experiments.CrossEntropyResult({args})'


def _fit_exponential_decay(x: Sequence[int], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an exponential model y = S * p**x using nonlinear least squares.

    Args:
        x: The x-values.
        y: The y-values.

    Returns:
        The result of calling `scipy.optimize.curve_fit`. This is a tuple of
        two arrays. The first array contains the fitted parameters, and the
        second array is their estimated covariance.
    """
    # Get initial guess by linear least squares with logarithm of model
    u = [a for a, b in zip(x, y) if b > 0]
    v = [np.log(b) for b in y if b > 0]
    fit = np.polynomial.polynomial.Polynomial.fit(u, v, 1).convert()
    p0 = np.exp(fit.coef)

    # Perform nonlinear least squares
    def f(a, S, p):
        return S * p**a

    return optimize.curve_fit(f, x, y, p0=p0)


@_compat.deprecated_class(
    deadline='v0.16', fix=('Use cirq.experiments.xeb_fitting.XEBCharacterizationResult instead')
)
@dataclasses.dataclass
class CrossEntropyResultDict(Mapping[Tuple['cirq.Qid', ...], CrossEntropyResult]):
    """Per-qubit-tuple results from cross-entropy benchmarking.

    Attributes:
        results: Dictionary from qubit tuple to cross-entropy benchmarking
            result for that tuple.
    """

    results: Dict[Tuple['cirq.Qid', ...], CrossEntropyResult]

    def _json_dict_(self) -> Dict[str, Any]:
        return {'results': list(self.results.items())}

    @classmethod
    def _from_json_dict_(
        cls, results: List[Tuple[List['cirq.Qid'], CrossEntropyResult]], **kwargs
    ) -> 'CrossEntropyResultDict':
        return cls(results={tuple(qubits): result for qubits, result in results})

    def __repr__(self) -> str:
        return f'cirq.experiments.CrossEntropyResultDict(results={self.results!r})'

    def __getitem__(self, key: Tuple['cirq.Qid', ...]) -> CrossEntropyResult:
        return self.results[key]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)
