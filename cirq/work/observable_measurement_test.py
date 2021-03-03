# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import cirq
import cirq.work as cw
from cirq.work.observable_measurement import (
    _with_parameterized_layers,
    _get_params_for_setting,
    _pad_setting,
    _subdivide_meas_specs,
)


def test_with_parameterized_layers():
    qs = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        [
            cirq.H.on_each(*qs),
            cirq.CZ(qs[0], qs[1]),
            cirq.CZ(qs[1], qs[2]),
        ]
    )
    circuit2 = _with_parameterized_layers(circuit, qubits=qs, needs_init_layer=False)
    assert circuit != circuit2
    assert len(circuit2) == 3 + 3  # 3 original, then X, Y, measure layer
    *_, xlayer, ylayer, measurelayer = circuit2.moments
    for op in xlayer.operations:
        assert isinstance(op.gate, cirq.XPowGate)
        assert op.gate.exponent.name.endswith('-Xf')
    for op in ylayer.operations:
        assert isinstance(op.gate, cirq.YPowGate)
        assert op.gate.exponent.name.endswith('-Yf')
    for op in measurelayer:
        assert isinstance(op.gate, cirq.MeasurementGate)

    circuit3 = _with_parameterized_layers(circuit, qubits=qs, needs_init_layer=True)
    assert circuit != circuit3
    assert circuit2 != circuit3
    assert len(circuit3) == 2 + 3 + 3
    xlayer, ylayer, *_ = circuit3.moments
    for op in xlayer.operations:
        assert isinstance(op.gate, cirq.XPowGate)
        assert op.gate.exponent.name.endswith('-Xi')
    for op in ylayer.operations:
        assert isinstance(op.gate, cirq.YPowGate)
        assert op.gate.exponent.name.endswith('-Yi')


def test_get_params_for_setting():
    qubits = cirq.LineQubit.range(3)
    a, b, c = qubits

    init_state = cirq.KET_PLUS(a) * cirq.KET_ZERO(b)
    observable = cirq.X(a) * cirq.Y(b)
    setting = cw.InitObsSetting(init_state=init_state, observable=observable)
    padded_setting = _pad_setting(setting, qubits=qubits)
    assert padded_setting.init_state == cirq.KET_PLUS(a) * cirq.KET_ZERO(b) * cirq.KET_ZERO(c)
    assert padded_setting.observable == cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    assert init_state == cirq.KET_PLUS(a) * cirq.KET_ZERO(b)
    assert observable == cirq.X(a) * cirq.Y(b)

    needs_init_layer = True
    with pytest.raises(ValueError):
        _get_params_for_setting(
            padded_setting,
            flips=[0, 0],
            qubits=qubits,
            needs_init_layer=needs_init_layer,
        )
    params = _get_params_for_setting(
        padded_setting,
        flips=[0, 0, 1],
        qubits=qubits,
        needs_init_layer=needs_init_layer,
    )
    assert all(
        x in params
        for x in [
            '0-Xf',
            '0-Yf',
            '1-Xf',
            '1-Yf',
            '2-Xf',
            '2-Yf',
            '0-Xi',
            '0-Yi',
            '1-Xi',
            '1-Yi',
            '2-Xi',
            '2-Yi',
        ]
    )

    circuit = cirq.Circuit(cirq.I.on_each(*qubits))
    circuit = _with_parameterized_layers(
        circuit,
        qubits=qubits,
        needs_init_layer=needs_init_layer,
    )
    circuit = circuit[:-1]  # remove measurement so we can compute <Z>
    psi = cirq.Simulator().simulate(circuit, param_resolver=params)
    ma = cirq.Z(a).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
    mb = cirq.Z(b).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
    mc = cirq.Z(c).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)

    np.testing.assert_allclose([ma, mb, mc], [1, 0, -1])


def test_params_and_settings():
    qubits = cirq.LineQubit.range(1)
    (q,) = qubits
    tests = [
        (cirq.KET_ZERO, cirq.Z, 1),
        (cirq.KET_ONE, cirq.Z, -1),
        (cirq.KET_PLUS, cirq.X, 1),
        (cirq.KET_MINUS, cirq.X, -1),
        (cirq.KET_IMAG, cirq.Y, 1),
        (cirq.KET_MINUS_IMAG, cirq.Y, -1),
        (cirq.KET_ZERO, cirq.Y, 0),
    ]

    for init, obs, coef in tests:
        setting = cw.InitObsSetting(
            init_state=init(q),
            observable=obs(q),
        )
        circuit = cirq.Circuit(cirq.I.on_each(*qubits))
        circuit = _with_parameterized_layers(circuit, qubits=qubits, needs_init_layer=True)
        params = _get_params_for_setting(
            setting, flips=[False], qubits=qubits, needs_init_layer=True
        )

        circuit = circuit[:-1]  # remove measurement so we can compute <Z>
        psi = cirq.Simulator().simulate(circuit, param_resolver=params)
        z = cirq.Z(q).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
        assert np.abs(coef - z) < 1e-2, f'{init} {obs} {coef}'


def test_subdivide_meas_specs():

    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    setting = cw.InitObsSetting(
        init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1)
    )
    meas_spec = cw._MeasurementSpec(
        max_setting=setting,
        circuit_params={
            'beta': 0.123,
            'gamma': 0.456,
        },
    )

    flippy_mspecs, repetitions = _subdivide_meas_specs(
        meas_specs=[meas_spec], repetitions=100_000, qubits=qubits, readout_symmetrization=True
    )
    fmspec1, fmspec2 = flippy_mspecs
    assert repetitions == 50_000
    assert fmspec1.meas_spec == meas_spec
    assert fmspec2.meas_spec == meas_spec
    assert np.all(fmspec2.flips)
    assert not np.any(fmspec1.flips)

    assert list(fmspec1.param_tuples()) == [
        ('0-Xf', 0),
        ('0-Yf', -0.5),
        ('0-Xi', 0),
        ('0-Yi', 0),
        ('1-Xf', 0.5),
        ('1-Yf', 0),
        ('1-Xi', 0),
        ('1-Yi', 0),
        ('beta', 0.123),
        ('gamma', 0.456),
    ]
