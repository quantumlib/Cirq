# Copyright 2021 The Cirq Developers
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

import numpy as np
import pytest

import cirq
from cirq.devices.noise_utils import (
    PHYSICAL_GATE_TAG,
)
from cirq.devices.thermal_noise_model import (
    _left_mul,
    _right_mul,
    _validate_rates,
    ThermalNoiseModel,
)


def test_helper_method_errors():
    with pytest.raises(ValueError, match='_left_mul only accepts square matrices'):
        _ = _left_mul(np.array([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match='_right_mul only accepts square matrices'):
        _ = _right_mul(np.array([[1, 2, 3], [4, 5, 6]]))

    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='qubits for rates inconsistent'):
        _validate_rates({q0, q1}, {q0: np.array([[0.01, 0.1], [0.02, 0.2]])})

    with pytest.raises(ValueError, match='qubits for rates inconsistent'):
        _validate_rates(
            {q0},
            {q0: np.array([[0.01, 0.1], [0.02, 0.2]]), q1: np.array([[0.03, 0.3], [0.04, 0.4]])},
        )

    with pytest.raises(ValueError, match='Invalid shape for rate matrix'):
        _validate_rates({q0}, {q0: np.array([[0.001, 0.01, 0.1], [0.002, 0.02, 0.2]])})


def test_create_thermal_noise_per_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.PhasedXZGate: 25.0}
    heat_rate_GHz = {q0: 1e-5, q1: 2e-5}
    cool_rate_GHz = {q0: 1e-4, q1: 2e-4}
    dephase_rate_GHz = {q0: 3e-4, q1: 4e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=dephase_rate_GHz,
    )
    assert model.gate_durations_ns == gate_durations
    assert model.require_physical_tag
    assert model.skip_measurements
    assert np.allclose(model.rate_matrix_GHz[q0], np.array([[0, 1e-4], [1e-5, 3e-4]]))
    assert np.allclose(model.rate_matrix_GHz[q1], np.array([[0, 2e-4], [2e-5, 4e-4]]))


def test_create_thermal_noise_mixed_type():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.PhasedXZGate: 25.0}
    heat_rate_GHz = None
    cool_rate_GHz = {q0: 1e-4, q1: 2e-4}
    dephase_rate_GHz = 3e-4
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=dephase_rate_GHz,
    )
    assert model.gate_durations_ns == gate_durations
    assert model.require_physical_tag
    assert model.skip_measurements
    assert np.allclose(model.rate_matrix_GHz[q0], np.array([[0, 1e-4], [0, 3e-4]]))
    assert np.allclose(model.rate_matrix_GHz[q1], np.array([[0, 2e-4], [0, 3e-4]]))


def test_incomplete_rates():
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.PhasedXZGate: 25.0}
    heat_rate_GHz = {q1: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
    )
    assert model.gate_durations_ns == gate_durations
    assert model.require_physical_tag
    assert model.skip_measurements
    assert np.allclose(model.rate_matrix_GHz[q0], np.array([[0, 1e-4], [0, 0]]))
    assert np.allclose(model.rate_matrix_GHz[q1], np.array([[0, 0], [1e-5, 0]]))


def test_noise_from_empty_moment():
    # Verify that a moment with no duration has no noise.
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {}
    heat_rate_GHz = {q1: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
        require_physical_tag=False,
        skip_measurements=False,
    )
    moment = cirq.Moment()
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]


def test_noise_from_zero_duration():
    # Verify that a moment with no duration has no noise.
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {}
    heat_rate_GHz = {q1: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
        require_physical_tag=False,
        skip_measurements=False,
    )
    moment = cirq.Moment(cirq.Z(q0), cirq.Z(q1))
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]


def test_noise_from_virtual_gates():
    # Verify that a moment with only virtual gates has no noise if
    # require_physical_tag is True.
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {cirq.ZPowGate: 25.0}
    heat_rate_GHz = {q1: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
        require_physical_tag=True,
        skip_measurements=False,
    )
    moment = cirq.Moment(cirq.Z(q0), cirq.Z(q1))
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]

    part_virtual_moment = cirq.Moment(cirq.Z(q0), cirq.Z(q1).with_tags(PHYSICAL_GATE_TAG))
    with pytest.raises(ValueError, match="all physical or all virtual"):
        _ = model.noisy_moment(part_virtual_moment, system_qubits=[q0, q1])

    model.require_physical_tag = False
    assert len(model.noisy_moment(moment, system_qubits=[q0, q1])) == 2


def test_noise_from_measurement():
    # Verify that a moment with only measurement gates has no noise if
    # skip_measurements is True.
    q0, q1 = cirq.LineQubit.range(2)
    gate_durations = {
        cirq.ZPowGate: 25.0,
        cirq.MeasurementGate: 4000.0,
    }
    heat_rate_GHz = {q1: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
        require_physical_tag=False,
        skip_measurements=True,
    )
    moment = cirq.Moment(cirq.measure(q0, q1, key='m'))
    assert model.noisy_moment(moment, system_qubits=[q0, q1]) == [moment]

    part_measure_moment = cirq.Moment(cirq.measure(q0, key='m'), cirq.Z(q1))
    assert len(model.noisy_moment(part_measure_moment, system_qubits=[q0, q1])) == 2

    model.skip_measurements = False
    assert len(model.noisy_moment(moment, system_qubits=[q0, q1])) == 2


def test_noisy_moment_one_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns={
            cirq.PhasedXZGate: 25.0,
            cirq.CZPowGate: 25.0,
        },
        heat_rate_GHz={q0: 1e-5, q1: 2e-5},
        cool_rate_GHz={q0: 1e-4, q1: 2e-4},
        dephase_rate_GHz={q0: 3e-4, q1: 4e-4},
        require_physical_tag=False,
    )
    gate = cirq.PhasedXZGate(x_exponent=1, z_exponent=0.5, axis_phase_exponent=0.25)
    moment = cirq.Moment(gate.on(q0))
    noisy_moment = model.noisy_moment(moment, system_qubits=[q0, q1])
    assert noisy_moment[0] == moment
    # Noise applies to both qubits, even if only one is acted upon.
    assert len(noisy_moment[1]) == 2
    noisy_choi = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[0]))
    assert np.allclose(
        noisy_choi,
        [
            [9.99750343e-01, 0, 0, 9.91164267e-01],
            [0, 2.49656565e-03, 0, 0],
            [0, 0, 2.49656565e-04, 0],
            [9.91164267e-01, 0, 0, 9.97503434e-01],
        ],
    )


def test_noise_from_wait():
    # Verify that wait-gate noise is duration-dependent.
    q0 = cirq.LineQubit(0)
    gate_durations = {cirq.ZPowGate: 25.0}
    heat_rate_GHz = {q0: 1e-5}
    cool_rate_GHz = {q0: 1e-4}
    model = ThermalNoiseModel(
        qubits={q0},
        gate_durations_ns=gate_durations,
        heat_rate_GHz=heat_rate_GHz,
        cool_rate_GHz=cool_rate_GHz,
        dephase_rate_GHz=None,
        require_physical_tag=False,
        skip_measurements=True,
    )
    moment = cirq.Moment(cirq.wait(q0, nanos=100))
    noisy_moment = model.noisy_moment(moment, system_qubits=[q0])
    assert noisy_moment[0] == moment
    assert len(noisy_moment[1]) == 1
    noisy_choi = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[0]))
    print(noisy_choi)
    assert np.allclose(
        noisy_choi,
        [
            [9.99005480e-01, 0, 0, 9.94515097e-01],
            [0, 9.94520111e-03, 0, 0],
            [0, 0, 9.94520111e-04, 0],
            [9.94515097e-01, 0, 0, 9.90054799e-01],
        ],
    )


def test_noisy_moment_two_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    model = ThermalNoiseModel(
        qubits={q0, q1},
        gate_durations_ns={
            cirq.PhasedXZGate: 25.0,
            cirq.CZPowGate: 25.0,
        },
        heat_rate_GHz={q0: 1e-5, q1: 2e-5},
        cool_rate_GHz={q0: 1e-4, q1: 2e-4},
        dephase_rate_GHz={q0: 3e-4, q1: 4e-4},
        require_physical_tag=False,
    )
    gate = cirq.CZ**0.5
    moment = cirq.Moment(gate.on(q0, q1))
    noisy_moment = model.noisy_moment(moment, system_qubits=[q0, q1])
    assert noisy_moment[0] == moment
    assert len(noisy_moment[1]) == 2
    noisy_choi_0 = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[0]))
    assert np.allclose(
        noisy_choi_0,
        [
            [9.99750343e-01, 0, 0, 9.91164267e-01],
            [0, 2.49656565e-03, 0, 0],
            [0, 0, 2.49656565e-04, 0],
            [9.91164267e-01, 0, 0, 9.97503434e-01],
        ],
    )
    noisy_choi_1 = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[1]))
    assert np.allclose(
        noisy_choi_1,
        [
            [9.99501372e-01, 0, 0, 9.87330937e-01],
            [0, 4.98627517e-03, 0, 0],
            [0, 0, 4.98627517e-04, 0],
            [9.87330937e-01, 0, 0, 9.95013725e-01],
        ],
    )
