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

from __future__ import annotations

from typing import AbstractSet, Any

import matplotlib.pyplot as plt
import numpy as np
import attrs
import sympy
import tunits as tu

import cirq

ValueOrSymbol = tu.Value | sympy.Symbol


@attrs.mutable
class FreqMap:
    """Object containing information about the step to a new analog Hamiltonian.

    Attributes:
        duration: duration of step
        qubit_freqs: dict describing qubit frequencies at end of step (None if idle)
        couplings: dict describing coupling rates at end of step
    """

    duration: ValueOrSymbol
    qubit_freqs: dict[str, ValueOrSymbol | None]
    couplings: dict[tuple[str, str], ValueOrSymbol]

    def _is_parameterized_(self) -> bool:
        def _is_parameterized_dict(dict_with_value: dict[str, ValueOrSymbol | None]) -> bool:
            return any(cirq.is_parameterized(v) for v in dict_with_value.values())

        return (
            cirq.is_parameterized(self.duration)
            or _is_parameterized_dict(self.qubit_freqs)
            or _is_parameterized_dict(self.couplings)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        def dict_param_name(dict_with_value: dict[Any, ValueOrSymbol | None]) -> AbstractSet[str]:
            return {v.name for v in dict_with_value.values() if cirq.is_parameterized(v)}

        return (
            cirq.parameter_names(self.duration)
            | dict_param_name(self.qubit_freqs)
            | dict_param_name(self.couplings)
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool
    ) -> FreqMap:
        def _direct_symbol_replacement(x, resolver: cirq.ParamResolver):
            if isinstance(x, sympy.Symbol):
                value = resolver.param_dict.get(x.name, "__NOT_FOUND__")
                if value == "__NOT_FOUND__":
                    value = resolver.param_dict.get(x, "__NOT_FOUND__")
                if value != "__NOT_FOUND__":
                    return value
                return x  # pragma: no cover
            return x

        resolver_ = cirq.ParamResolver(resolver)
        return FreqMap(
            duration=_direct_symbol_replacement(self.duration, resolver_),
            qubit_freqs={
                k: _direct_symbol_replacement(v, resolver_) for k, v in self.qubit_freqs.items()
            },
            couplings={
                k: _direct_symbol_replacement(v, resolver_) for k, v in self.couplings.items()
            },
        )


class AnalogTrajectory:
    """Class for handling qubit frequency and coupling trajectories that
    define analog experiments. The class is defined using a sparse_trajectory,
    which contains time durations of each Hamiltonian ramp element and the
    corresponding qubit frequencies and couplings (unassigned qubits and/or
    couplers are left unchanged).
    """

    def __init__(
        self, *, full_trajectory: list[FreqMap], qubits: list[str], pairs: list[tuple[str, str]]
    ):
        self.full_trajectory = full_trajectory
        self.qubits = qubits
        self.pairs = pairs

    @classmethod
    def from_sparse_trajecotry(
        cls,
        sparse_trajectory: list[
            tuple[tu.Value, dict[str, ValueOrSymbol | None], dict[tuple[str, str], ValueOrSymbol]],
        ],
        qubits: list[str] | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ):
        """Construct AnalogTrajectory from sparse trajectory.

        Args:
            sparse_trajectory: A list of tuples, where each tuple defines a segment of `FreqMap`
                and contains three elements: (duration, qubit_freqs, coupling_strengths).
                `duration` is a tunits value, `qubit_freqs` is a dictionary mapping qubit strings
                to detuning frequencies, and `coupling_strengths` is a dictionary mapping qubit
                pairs to their coupling strength. This format is considered "sparse" because each
                tuple does not need to fully specify all qubits and coupling pairs; any missing
                detuning frequency or coupling strength will be set to the same value as the
                previous value in the list.
            qubits: The qubits in interest. If not provided, automatically parsed from trajectory.
            pairs: The pairs in interest. If not provided, automatically parsed from trajectory.
        """
        if qubits is None or pairs is None:
            qubits_in_traj: list[str] = []
            pairs_in_traj: list[tuple[str, str]] = []
            for _, q, p in sparse_trajectory:
                qubits_in_traj.extend(q.keys())
                pairs_in_traj.extend(p.keys())
            qubits = list(set(qubits_in_traj))
            pairs = list(set(pairs_in_traj))

        full_trajectory: list[FreqMap] = []
        init_qubit_freq_dict: dict[str, tu.Value | None] = {q: None for q in qubits}
        init_g_dict: dict[tuple[str, str], tu.Value] = {p: 0 * tu.MHz for p in pairs}
        full_trajectory.append(FreqMap(0 * tu.ns, init_qubit_freq_dict, init_g_dict))

        for dt, qubit_freq_dict, g_dict in sparse_trajectory:
            # If no freq provided, set equal to previous
            new_qubit_freq_dict = {
                q: qubit_freq_dict.get(q, full_trajectory[-1].qubit_freqs.get(q)) for q in qubits
            }
            # If no g provided, set equal to previous
            new_g_dict: dict[tuple[str, str], tu.Value] = {
                p: g_dict.get(p, full_trajectory[-1].couplings.get(p)) for p in pairs
            }

            full_trajectory.append(FreqMap(dt, new_qubit_freq_dict, new_g_dict))
        return cls(full_trajectory=full_trajectory, qubits=qubits, pairs=pairs)

    def get_full_trajectory_with_resolved_idles(
        self, idle_freq_map: dict[str, tu.Value]
    ) -> list[FreqMap]:
        """Insert idle frequencies instead of None in trajectory."""

        resolved_trajectory: list[FreqMap] = []
        for dt, qubit_freq_dict, g_dict in self.full_trajectory:
            resolved_qubit_freq_dict = {
                q: idle_freq_map[q] if f is None else f for q, f in qubit_freq_dict.items()
            }
            resolved_trajectory.append(FreqMap(dt, resolved_qubit_freq_dict, g_dict))  # type: ignore
        return resolved_trajectory

    # TODO make this plot better.
    def plot(
        self,
        idle_freq_map: dict[str, tu.Value] | None = None,
        default_idle_freq: tu.Value = 6.5 * tu.GHz,
        resolver: cirq.ParamResolverOrSimilarType | None = None,
    ):
        if idle_freq_map is None:
            idle_freq_map = {q: default_idle_freq for q in self.qubits}
        full_trajectory_resolved = cirq.resolve_parameters(
            self.get_full_trajectory_with_resolved_idles(idle_freq_map), resolver
        )

        plt.figure(figsize=(10, 4))
        times = np.cumsum([step.duration[tu.ns] for step in full_trajectory_resolved])
        ylabels = ["Qubit freq. (GHz)", "Coupling (MHz)"]

        plt.subplot(1, 2, 1)
        for qubit_agent in self.qubits:
            plt.plot(
                times,
                [step.qubit_freqs[qubit_agent][tu.GHz] for step in full_trajectory_resolved],
                label=qubit_agent,
            )
        plt.subplot(1, 2, 2)
        for pair_agent in self.pairs:
            plt.plot(
                times,
                [step.couplings[pair_agent][tu.MHz] for step in full_trajectory_resolved],
                label=pair_agent,
            )

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.legend()
            plt.xlabel("Time (ns)")
            plt.ylabel(ylabels[i])
            plt.tight_layout()
