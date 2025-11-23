# Copyright 2025 The Cirq Developers
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

from __future__ import annotations

from collections.abc import Set
from typing import TYPE_CHECKING

import attrs
import matplotlib.pyplot as plt
import numpy as np
import tunits as tu

import cirq
from cirq_google.study import symbol_util as su

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cirq_google.ops import coupler as cgc


@attrs.mutable
class FrequencyMap:
    """Object containing information about the step to a new analog Hamiltonian.

    Attributes:
        duration: duration of step
        qubit_freqs: dict describing qubit frequencies at end of step (None if idle)
        couplings: dict describing coupling rates at end of step
        is_wait_step: a bool indicating only wait gate should be added.
    """

    duration: su.ValueOrSymbol
    qubit_freqs: dict[cirq.Qid, su.ValueOrSymbol | None]
    couplings: dict[cgc.Coupler, su.ValueOrSymbol]
    is_wait_step: bool

    def _is_parameterized_(self) -> bool:
        return (
            cirq.is_parameterized(self.duration)
            or su.is_parameterized_dict(self.qubit_freqs)
            or su.is_parameterized_dict(self.couplings)
        )

    def _parameter_names_(self) -> Set[str]:
        return (
            cirq.parameter_names(self.duration)
            | su.dict_param_name(self.qubit_freqs)
            | su.dict_param_name(self.couplings)
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool
    ) -> FrequencyMap:
        resolver_ = cirq.ParamResolver(resolver)
        return FrequencyMap(
            duration=su.direct_symbol_replacement(self.duration, resolver_),
            qubit_freqs={
                k: su.direct_symbol_replacement(v, resolver_) for k, v in self.qubit_freqs.items()
            },
            couplings={
                k: su.direct_symbol_replacement(v, resolver_) for k, v in self.couplings.items()
            },
            is_wait_step=self.is_wait_step,
        )


class AnalogTrajectory:
    """Class for handling qubit frequency and coupling trajectories that
    define analog experiments. The class is defined using a sparse_trajectory,
    which contains time durations of each Hamiltonian ramp element and the
    corresponding qubit frequencies and couplings (unassigned qubits and/or
    couplers are left unchanged).
    """

    def __init__(
        self,
        *,
        full_trajectory: list[FrequencyMap],
        qubits: list[cirq.Qid],
        couplers: list[cgc.Coupler],
    ):
        self.full_trajectory = full_trajectory
        self.qubits = qubits
        self.couplers = couplers

    @classmethod
    def from_sparse_trajectory(
        cls,
        sparse_trajectory: list[
            tuple[
                tu.Value,
                dict[cirq.Qid, su.ValueOrSymbol | None],
                dict[cgc.Coupler, su.ValueOrSymbol],
            ],
        ],
        qubits: list[cirq.Qid] | None = None,
        couplers: list[cgc.Coupler] | None = None,
    ):
        """Construct AnalogTrajectory from sparse trajectory.

        Args:
            sparse_trajectory: A list of tuples, where each tuple defines a `FrequencyMap`
                and contains three elements: (duration, qubit_freqs, coupling_strengths).
                `duration` is a tunits value, `qubit_freqs` is a dictionary mapping cirq qubits
                to detuning frequencies, and `coupling_strengths` is a dictionary mapping
                couplers to their coupling strength. This format is considered "sparse" because each
                tuple does not need to fully specify all qubits and coupling pairs; any missing
                detuning frequency or coupling strength will be set to the same value as the
                previous value in the list.
            qubits: The qubits in interest. If not provided, automatically parsed from trajectory.
            couplers: The couplers in interest. If not provided, auto. parsed from trajectory.
        """
        if qubits is None or couplers is None:
            qubits_in_traj: list[cirq.Qid] = []
            couplers_in_traj: list[cgc.Coupler] = []
            for _, q, p in sparse_trajectory:
                qubits_in_traj.extend(q.keys())
                couplers_in_traj.extend(p.keys())
            qubits = list(set(qubits_in_traj))
            couplers = list(set(couplers_in_traj))

        full_trajectory: list[FrequencyMap] = []
        init_qubit_freq_dict: dict[cirq.Qid, tu.Value | None] = {q: None for q in qubits}
        init_g_dict: dict[cgc.Coupler, tu.Value] = {c: 0 * tu.MHz for c in couplers}
        full_trajectory.append(FrequencyMap(0 * tu.ns, init_qubit_freq_dict, init_g_dict, False))

        for dt, qubit_freq_dict, g_dict in sparse_trajectory:
            # When both qubit_freq_dict and g_dict is empty, it is a wait step.
            is_wait_step = not (qubit_freq_dict or g_dict)
            # If no freq provided, set equal to previous
            new_qubit_freq_dict = {
                q: qubit_freq_dict.get(q, full_trajectory[-1].qubit_freqs.get(q)) for q in qubits
            }
            # If no g provided, set equal to previous
            new_g_dict: dict[cgc.Coupler, tu.Value] = {
                c: g_dict.get(c, full_trajectory[-1].couplings.get(c)) for c in couplers  # type: ignore[misc]
            }

            full_trajectory.append(FrequencyMap(dt, new_qubit_freq_dict, new_g_dict, is_wait_step))
        return cls(full_trajectory=full_trajectory, qubits=qubits, couplers=couplers)

    def get_full_trajectory_with_resolved_idles(
        self, idle_freq_map: dict[cirq.Qid, tu.Value]
    ) -> list[FrequencyMap]:
        """Insert idle frequencies instead of None in trajectory."""

        resolved_trajectory: list[FrequencyMap] = []
        for freq_map in self.full_trajectory:
            resolved_qubit_freqs = {
                q: idle_freq_map[q] if f is None else f for q, f in freq_map.qubit_freqs.items()
            }
            resolved_trajectory.append(attrs.evolve(freq_map, qubit_freqs=resolved_qubit_freqs))
        return resolved_trajectory

    def plot(
        self,
        idle_freq_map: dict[cirq.Qid, tu.Value] | None = None,
        resolver: cirq.ParamResolverOrSimilarType | None = None,
        axes: tuple[Axes, Axes] | None = None,
    ) -> tuple[Axes, Axes]:
        if idle_freq_map is None:
            # Because we use relative frequencies and we do not expose the idle frequencies,
            # we randomly assign idle frequencies for plotting purposes only.
            idle_freq_map = {q: np.random.randn() * 50 * tu.MHz for q in self.qubits}
        else:  # pragma: no cover
            for q in self.qubits:
                if q not in idle_freq_map:  # Fill in missing idle freqs
                    idle_freq_map[q] = np.random.randn() * 50 * tu.MHz

        full_trajectory_resolved = cirq.resolve_parameters(
            self.get_full_trajectory_with_resolved_idles(idle_freq_map), resolver
        )
        unresolved_param_names = set().union(
            *[cirq.parameter_names(freq_map) for freq_map in full_trajectory_resolved]
        )
        if unresolved_param_names:
            raise ValueError(f"There are some parameters {unresolved_param_names} not resolved.")

        times = np.cumsum([step.duration[tu.ns] for step in full_trajectory_resolved])

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 4))

        for qubit in self.qubits:
            axes[0].plot(
                times,
                [step.qubit_freqs[qubit][tu.GHz] for step in full_trajectory_resolved],  # type: ignore[index]
                label=qubit,
            )
        for coupler in self.couplers:
            axes[1].plot(
                times,
                [step.couplings[coupler][tu.MHz] for step in full_trajectory_resolved],
                label=coupler,
            )

        for ax, ylabel in zip(axes, ["Qubit freq. (GHz)", "Coupling (MHz)"]):
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(ylabel)
            ax.legend()
        plt.tight_layout()
        return axes
