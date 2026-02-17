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
from collections.abc import Sequence
from typing import cast

import networkx as nx
import numpy as np
import scipy
import tunits as tu

import cirq
import cirq.contrib.routing as ccr
import cirq.experiments.random_quantum_circuit_generation as rcg
from cirq_google.experimental.analog_experiments import analog_trajectory_util as atu
from cirq_google.ops import analog_detune_gates as adg, coupler as cgc, wait_gate as wg
from cirq_google.study import symbol_util as su


def _get_neighbor_freqs(
    coupler: cgc.Coupler, qubit_freq_dict: dict[cirq.Qid, su.ValueOrSymbol | None]
) -> tuple[su.ValueOrSymbol | None, su.ValueOrSymbol | None]:
    """Get neighbor freqs from qubit_freq_dict given the coupler."""
    sorted_pair = sorted(coupler.qubits)
    return (qubit_freq_dict[sorted_pair[0]], qubit_freq_dict[sorted_pair[1]])


@functools.cache
def _to_grid_qubit(qubit_name: str) -> cirq.GridQubit:
    match = re.compile(r"^q(\d+)_(\d+)$").match(qubit_name)
    if match is None:
        raise ValueError(f"Invalid qubit name format: '{qubit_name}'. Expected 'q<row>_<col>'.")
    return cirq.GridQubit(int(match[1]), int(match[2]))


def _coupler_name(coupler: cgc.Coupler) -> str:
    q1, q2 = sorted(coupler.qubits)
    return f"c_q{q1.row}_{q1.col}_q{q2.row}_{q2.col}"  # type: ignore[attr-defined]


def _get_neighbor_coupler_freqs(
    qubit: cirq.Qid, coupler_g_dict: dict[cgc.Coupler, su.ValueOrSymbol]
) -> dict[str, su.ValueOrSymbol]:
    """Get neighbor coupler coupling strength g given qubit name."""
    return {
        _coupler_name(coupler): g
        for coupler, g in coupler_g_dict.items()
        if qubit in coupler.qubits
    }


class GenericAnalogCircuitBuilder:
    """Class for making arbitrary analog circuits. The circuit is defined by an
    AnalogTrajectory object. The class constructs the circuit from AnalogDetune
    pulses, which automatically calculate the necessary bias amps to both qubits
    and couplers, using tu.Values from analog calibration whenever available.

    Attributes:
        trajectory: AnalogTrajectory object defining the circuit.
        g_ramp_shaping: coupling ramps are shaped according to ramp_shape_exp if True.
        ramp_shape_exp: exponent of g_ramp (g proportional to t^ramp_shape_exp).
        interpolate_coupling_cal: interpolates between calibrated coupling tu.Values if True.
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
                targets = self.trajectory.qubits
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
                ).on(q)
            )
        coupler_gates = []
        for c, g_max in freq_map.couplings.items():
            # Currently skipping the step if these are the same.
            # However, change in neighbor qubit freq could potentially change coupler amp
            if g_max == prev_freq_map.couplings[c]:
                continue

            coupler_gates.append(
                adg.AnalogDetuneCouplerOnly(
                    length=freq_map.duration,
                    w=freq_map.duration,
                    g_0=prev_freq_map.couplings[c],
                    g_max=g_max,
                    g_ramp_exponent=self.ramp_shape_exp,
                    neighbor_qubits_freq=_get_neighbor_freqs(c, freq_map.qubit_freqs),
                    prev_neighbor_qubits_freq=_get_neighbor_freqs(c, prev_freq_map.qubit_freqs),
                    interpolate_coupling_cal=self.interpolate_coupling_cal,
                ).on(c)
            )

        return cirq.Moment(qubit_gates + coupler_gates)


class AnalogSimulationCircuitBuilder:
    """Class for making arbitrary analog circuits for the purpose of simulating them.
    The circuit is defined by an AnalogTrajectory object.

    Attributes:
        trajectory: AnalogTrajectory object defining the circuit.
        g_ramp_shaping: coupling ramps are shaped according to ramp_shape_exp if True.
        ramp_shape_exp: exponent of g_ramp (g proportional to t^ramp_shape_exp).
        interpolate_coupling_cal: interpolates between calibrated coupling tu.Values if True.
        linear_qubit_ramp: if True, the qubit ramp is linear. if false, a cosine shaped
            ramp is used.
    """

    def __init__(
        self,
        trajectory: atu.AnalogTrajectory,
        g_ramp_shaping: bool = True,
        ramp_shape_exp: int = 1,
        interpolate_coupling_cal: bool = False,
        linear_qubit_ramp: bool = True,
    ):

        if ramp_shape_exp != 1 or not linear_qubit_ramp:
            raise NotImplementedError("Only linear ramps are currently supported.")

        self.trajectory = trajectory
        self.g_ramp_shaping = g_ramp_shaping
        self.ramp_shape_exp = ramp_shape_exp
        self.interpolate_coupling_cal = interpolate_coupling_cal
        self.linear_qubit_ramp = linear_qubit_ramp

    def _get_interpolators(
        self, idle_freq_map: dict[cirq.Qid, tu.Value] | None
    ) -> tuple[dict, float]:
        """Get interpolators for the qubit frequencies and coupling strengths that can be evaluated
        at any time.

        Args:
            idle_freq_map: A map from qubits to their idle frequencies. If None, assume idles are 0.

        Returns:
            * A map from qubits or couplers to functions from time to the qubit frequency or
            coupling strength.
            * The maximum time in nanoseconds for which these functions can be evaluated.
        """

        # resolve the idles
        idle_freq_map_resolved = (
            dict.fromkeys(self.trajectory.qubits, 0 * tu.GHz)
            if idle_freq_map is None
            else idle_freq_map
        )
        full_trajectory = self.trajectory.get_full_trajectory_with_resolved_idles(
            idle_freq_map_resolved
        )

        t_traj = [_.duration[tu.ns] for _ in self.trajectory.full_trajectory]
        for t in t_traj[1:]:
            if not t > 0:
                raise ValueError("Trajectory times should be positive.")  # pragma: no cover

        t = np.cumsum(t_traj)
        interpolators = {}
        for qubit in self.trajectory.qubits:
            f = [cast(tu.Value, _.qubit_freqs[qubit])[tu.GHz] for _ in full_trajectory]
            interpolators[qubit] = scipy.interpolate.interp1d(t, f)
        for coupler in self.trajectory.couplers:
            f = [_.couplings[coupler][tu.GHz] for _ in full_trajectory]
            interpolators[coupler] = scipy.interpolate.interp1d(t, f)
        return interpolators, t[-1]

    def _make_first_order_circuit(
        self,
        interpolators: dict,
        dt_ns: float,
        num_steps: int,
        device_graph: nx.Graph,
        interaction_pattern: Sequence[rcg.GridInteractionLayer],
    ) -> cirq.Circuit:
        moments = []
        for step in range(num_steps):
            # single-qubit moment:
            frequency_GHz = {q: interpolators[q](step * dt_ns) for q in self.trajectory.qubits}
            moments.append(
                cirq.Moment((cirq.Z ** (-2 * f * dt_ns))(q) for q, f in frequency_GHz.items())
            )

            # two qubit moments:
            coupling_GHz = {c: interpolators[c](step * dt_ns) for c in self.trajectory.couplers}
            for layer in interaction_pattern:
                pairs = list(rcg._get_active_pairs(device_graph, layer))
                if len(pairs) > 0:
                    moments.append(
                        cirq.Moment(
                            (cirq.ISWAP ** (-4 * coupling_GHz[cgc.Coupler(*pair)] * dt_ns)).on(
                                *pair
                            )
                            for pair in pairs
                        )
                    )
        return cirq.Circuit.from_moments(*moments)

    def _make_second_order_circuit(
        self,
        interpolators: dict,
        dt_ns: float,
        num_steps: int,
        device_graph: nx.Graph,
        interaction_pattern: Sequence[rcg.GridInteractionLayer],
    ) -> cirq.Circuit:
        moments = []
        for step in range(num_steps):
            # single-qubit moment:
            frequency_GHz = {q: interpolators[q](step * dt_ns) for q in self.trajectory.qubits}
            single_qubit_moment = cirq.Moment(
                (cirq.Z ** (-f * dt_ns))(q) for q, f in frequency_GHz.items()
            )  # dt/2

            # two qubit moments:
            coupling_GHz = {c: interpolators[c](step * dt_ns) for c in self.trajectory.couplers}

            # two qubit moments:
            two_qubit_moments_except_last = []
            for layer in interaction_pattern[:-1]:
                pairs = list(rcg._get_active_pairs(device_graph, layer))
                if len(pairs) > 0:
                    two_qubit_moments_except_last.append(
                        cirq.Moment(
                            (cirq.ISWAP ** (-2 * coupling_GHz[cgc.Coupler(*pair)] * dt_ns)).on(
                                *pair
                            )
                            for pair in pairs
                        )
                    )  # dt/2

            last_two_qubit_moment = []
            pairs = list(rcg._get_active_pairs(device_graph, interaction_pattern[-1]))
            if len(pairs) > 0:
                last_two_qubit_moment.append(
                    cirq.Moment(
                        (cirq.ISWAP ** (-4 * coupling_GHz[cgc.Coupler(*pair)] * dt_ns)).on(*pair)
                        for pair in pairs
                    )
                )  # dt

            moments.extend(
                [
                    single_qubit_moment,
                    *two_qubit_moments_except_last,
                    *last_two_qubit_moment,
                    *two_qubit_moments_except_last[::-1],
                    single_qubit_moment,
                ]
            )
        return cirq.Circuit.from_moments(*moments)

    def make_circuit(
        self,
        trotter_step: tu.Value,
        interaction_pattern: Sequence[rcg.GridInteractionLayer] = rcg.HALF_GRID_STAGGERED_PATTERN,
        second_order: bool = False,
        idle_freq_map: dict[cirq.Qid, tu.Value] | None = None,
    ) -> cirq.Circuit:
        r"""Create a Cirq circuit for simulating an analog experiment.

        The resulting circuit can be put into a larger circuit containing digital
        gates and measurements and can be simulated using cirq simulators such as
        cirq.Simulator or qsimcirq.QSimSimulator.

        Coupling terms are modeled as $e^{-2\pi i g dt(X_i X_j + Y_i Y_j)/2}$, which is
        `cirq.ISWAP**(-4*g*dt)`
        (see https://quantumai.google/reference/python/cirq/ISwapPowGate).

        Single-qubit terms are modeled as $e^{-2\pi i f dt \hat n}$, which, up to a global
        phase is $e^{\pi i f dt Z}$, i.e. `cirq.Z**(-2*f*dt)`
        (see https://quantumai.google/reference/python/cirq/ZPowGate).

        Current limitations:
        * For now, phases should not be expected to match those in the actual experiment.
            Experimental phases need to be characterized.
        * Uses a simplified XY model Hamiltonian instead of the full device Hamiltonian,
        which includes higher levels and higher-order nonlocal interactions.
        * Ramps faster than 2-3 nanoseconds are not possible on hardware; filters
        smooth out such fast ramps but are not simulated here.
        * Sympy symbols are not supported. Please resolve all parameters.

        Args:
            trotter_step: The Trotter step size used for simulation.
            interaction_pattern: The pattern of two-qubit gates to use for the simulation.
                Shouldn't matter as long as the Trotter step size is sufficiently small.
            second_order: Whether to use a second-order Trotter approximation
                (otherwise use first-order).
            idle_freq_map: The qubit idle frequencies. If not provided, set to 0.

        Returns:
            A circuit that can be used with a simulator.
        """

        # next, get interpolators
        interpolators, t_max_ns = self._get_interpolators(idle_freq_map)

        # assert that this is an integer number of Trotter steps
        dt_ns = trotter_step[tu.ns]
        num_steps = int(np.round(t_max_ns / dt_ns))
        if not np.isclose(num_steps * dt_ns, t_max_ns, atol=1e-5):
            raise ValueError(  # pragma: no cover
                "Please pick a Trotter step that divides the total time, "  # pragma: no cover
                f"{t_max_ns} ns"  # pragma: no cover
            )  # pragma: no cover

        # get the device graph
        grid_qubit_list = []
        for qubit in self.trajectory.qubits:
            assert type(qubit) == cirq.GridQubit, "Qubits must be cirq.GridQubit"
            grid_qubit_list.append(qubit)
        device_graph = ccr.gridqubits_to_graph_device(grid_qubit_list)

        # make the circuit

        return (
            self._make_second_order_circuit(
                interpolators, dt_ns, num_steps, device_graph, interaction_pattern
            )
            if second_order
            else self._make_first_order_circuit(
                interpolators, dt_ns, num_steps, device_graph, interaction_pattern
            )
        )
