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

import functools
import re

import cirq
from cirq_google.experimental.analog_experiments import analog_trajectory_util as atu
from cirq_google.ops import analog_detune_gates as adg, wait_gate as wg
from cirq_google.study import symbol_util as su


def _get_neighbor_freqs(
    qubit_pair: tuple[str, str], qubit_freq_dict: dict[str, su.ValueOrSymbol | None]
) -> tuple[su.ValueOrSymbol | None, su.ValueOrSymbol | None]:
    """Get neighbor freqs from qubit_freq_dict given the pair."""
    sorted_pair = sorted(qubit_pair, key=_to_grid_qubit)
    return (qubit_freq_dict[sorted_pair[0]], qubit_freq_dict[sorted_pair[1]])


@functools.cache
def _to_grid_qubit(qubit_name: str) -> cirq.GridQubit:
    match = re.compile(r"^q(\d+)_(\d+)$").match(qubit_name)
    if match is None:
        raise ValueError(f"Invalid qubit name format: '{qubit_name}'. Expected 'q<row>_<col>'.")
    return cirq.GridQubit(int(match[1]), int(match[2]))


def _coupler_name_from_qubit_pair(qubit_pair: tuple[str, str]) -> str:
    sorted_pair = sorted(qubit_pair, key=_to_grid_qubit)
    return f"c_{sorted_pair[0]}_{sorted_pair[1]}"


def _get_neighbor_coupler_freqs(
    qubit_name: str, coupler_g_dict: dict[tuple[str, str], su.ValueOrSymbol]
) -> dict[str, su.ValueOrSymbol]:
    """Get neighbor coupler coupling strength g given qubit name."""
    return {
        _coupler_name_from_qubit_pair(pair): g
        for pair, g in coupler_g_dict.items()
        if qubit_name in pair
    }


class GenericAnalogCircuitBuilder:
    """Class for making arbitrary analog circuits. The circuit is defined by an
    AnalogTrajectory object. The class constructs the circuit from AnalogDetune
    pulses, which automatically calculate the necessary bias amps to both qubits
    and couplers, using tu.Values from analog calibration whenever available.

    Attributes:
        trajectory: AnalogTrajectory object defining the circuit
        g_ramp_shaping: coupling ramps are shaped according to ramp_shape_exp if True
        qubits: list of qubits in the circuit
        pairs: list of couplers in the circuit
        ramp_shape_exp: exponent of g_ramp (g proportional to t^ramp_shape_exp)
        interpolate_coupling_cal: interpolates between calibrated coupling tu.Values if True
        linear_qubit_ramp: if True, the qubit ramp is linear. if false, a cosine shaped
            ramp is used.
    """

    def __init__(
        self,
        trajectory: atu.AnalogTrajectory,
        g_ramp_shaping: bool = False,
        ramp_shape_exp: int = 1,
        interpolate_coupling_cal: bool = False,
        linear_qubit_ramp: bool = True,
    ):
        self.trajectory = trajectory
        self.g_ramp_shaping = g_ramp_shaping
        self.ramp_shape_exp = ramp_shape_exp
        self.interpolate_coupling_cal = interpolate_coupling_cal
        self.linear_qubit_ramp = linear_qubit_ramp

    def make_circuit(self) -> cirq.Circuit:
        """Assemble moments described in trajectory."""
        prev_freq_map = self.trajectory.full_trajectory[0]
        moments = []
        for freq_map in self.trajectory.full_trajectory[1:]:
            if freq_map.is_wait_step:
                targets = [_to_grid_qubit(q) for q in self.trajectory.qubits]
                wait_gate = wg.WaitGateWithUnit(
                    freq_map.duration, qid_shape=cirq.qid_shape(targets)
                )
                moment = cirq.Moment(wait_gate.on(*targets))
            else:
                moment = self.make_one_moment(freq_map, prev_freq_map)
            moments.append(moment)
            prev_freq_map = freq_map

        return cirq.Circuit.from_moments(*moments)

    def make_one_moment(
        self, freq_map: atu.FrequencyMap, prev_freq_map: atu.FrequencyMap
    ) -> cirq.Moment:
        """Make one moment of analog detune qubit and coupler gates given freqs."""
        qubit_gates = []
        for q, freq in freq_map.qubit_freqs.items():
            qubit_gates.append(
                adg.AnalogDetuneQubit(
                    length=freq_map.duration,
                    w=freq_map.duration,
                    target_freq=freq,
                    prev_freq=prev_freq_map.qubit_freqs.get(q),
                    neighbor_coupler_g_dict=_get_neighbor_coupler_freqs(q, freq_map.couplings),
                    prev_neighbor_coupler_g_dict=_get_neighbor_coupler_freqs(
                        q, prev_freq_map.couplings
                    ),
                    linear_rise=self.linear_qubit_ramp,
                ).on(_to_grid_qubit(q))
            )
        coupler_gates = []
        for p, g_max in freq_map.couplings.items():
            # Currently skipping the step if these are the same.
            # However, change in neighbor qubit freq could potentially change coupler amp
            if g_max == prev_freq_map.couplings[p]:
                continue

            coupler_gates.append(
                adg.AnalogDetuneCouplerOnly(
                    length=freq_map.duration,
                    w=freq_map.duration,
                    g_0=prev_freq_map.couplings[p],
                    g_max=g_max,
                    g_ramp_exponent=self.ramp_shape_exp,
                    neighbor_qubits_freq=_get_neighbor_freqs(p, freq_map.qubit_freqs),
                    prev_neighbor_qubits_freq=_get_neighbor_freqs(p, prev_freq_map.qubit_freqs),
                    interpolate_coupling_cal=self.interpolate_coupling_cal,
                ).on(*sorted([_to_grid_qubit(p[0]), _to_grid_qubit(p[1])]))
            )

        return cirq.Moment(qubit_gates + coupler_gates)
