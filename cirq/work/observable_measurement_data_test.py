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
import datetime

import numpy as np
import pytest

import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import _get_real_coef, _obs_vals_from_measurements, \
    _stats_from_measurements
from cirq.work.observable_settings import _MeasurementSpec


def test_get_real_coef():
    q0 = cirq.LineQubit(0)
    assert _get_real_coef(cirq.Z(q0) * 2) == 2
    assert _get_real_coef(cirq.Z(q0) * complex(2.0)) == 2
    with pytest.raises(ValueError):
        _get_real_coef(cirq.Z(q0) * 2.j)


def test_obs_vals_from_measurements():
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    vals = _obs_vals_from_measurements(bitstrings, qubit_to_index, obs)
    should_be = [10, -10, -10, 10]
    np.testing.assert_equal(vals, should_be)


def test_stats_from_measurements():
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    mean, err = _stats_from_measurements(bitstrings, qubit_to_index, obs)

    # The mean is zero since our bitstrings have balanced even- and odd-
    # partiy cases.
    assert mean == 0

    # Since we multiplied our observable by 10, the standard deviation is
    # 10 [each obs val deviates by 10]. The variance is 10**2 and the
    # squared-standard-error-of-the-mean can be found by dividing by the
    # number of samples.
    assert err == 10 ** 2 / 4


def test_observable_measured_result():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    omr = cw.ObservableMeasuredResult(
        setting=cw.InitObsSetting(
            init_state=cirq.Z(a) * cirq.Z(b),
            observable=cirq.Y(a) * cirq.Y(b),
        ),
        mean=0,
        variance=5 ** 2,
        repetitions=4,
        circuit_params={}
    )
    assert omr.stddev == 5
    assert omr.observable == cirq.Y(a) * cirq.Y(b)
    assert omr.init_state == cirq.Z(a) * cirq.Z(b)


@pytest.fixture()
def example_bsa() -> 'cw.BitstringAccumulator':
    """Test fixture to create an (empty) example BitstringAccumulator"""
    q0, q1 = cirq.LineQubit.range(2)
    setting = cw.InitObsSetting(init_state=cirq.KET_ZERO(q0) *
                                           cirq.KET_ZERO(q1),
                                observable=cirq.X(q0) * cirq.Y(q1))
    meas_spec = _MeasurementSpec(max_setting=setting,
                                 circuit_params={
                                     'beta': 0.123,
                                     'gamma': 0.456,
                                 })
    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[
            setting,
            cw.InitObsSetting(init_state=setting.init_state, observable=cirq.X(q0)),
            cw.InitObsSetting(init_state=setting.init_state, observable=cirq.Y(q1)),
        ],
        qubit_to_index={q0: 0, q1: 1},
    )
    return bsa


def test_bitstring_accumulator(example_bsa):
    # test initialization
    assert example_bsa.bitstrings.shape == (0, 2)
    assert example_bsa.chunksizes.shape == (0,)
    assert example_bsa.timestamps.shape == (0,)

    # test consume_results
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    example_bsa.consume_results(bitstrings)
    assert example_bsa.bitstrings.shape == (4, 2)
    assert example_bsa.chunksizes.shape == (1,)
    assert example_bsa.timestamps.shape == (1,)
    assert example_bsa.n_repetitions == 4

    # test results
    results = list(example_bsa.results)
    assert len(results) == 3
    for r in results:
        assert r.repetitions == 4

    # test records
    for r in example_bsa.records:
        assert isinstance(r, dict)
        assert 'repetitions' in r
        assert r['repetitions'] == 4


def test_bitstring_accumulator_equality():
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    setting = cw.InitObsSetting(init_state=cirq.Z(a) * cirq.Z(b), observable=obs)
    meas_spec = _MeasurementSpec(setting, {})

    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )

    bsa2 = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )
    assert bsa == bsa2

    timestamps = np.asarray([datetime.datetime.now()])
    bsa3 = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )
    assert bsa != bsa3


def test_bitstring_accumulator_stats():
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    chunksizes = [4]
    timestamps = [datetime.datetime.now()]
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    settings = list(cw.observables_to_settings([
        cirq.Z(a) * cirq.Z(b) * 10,
        cirq.Z(a) * 10,
        cirq.Z(b) * 10
    ], qubits=[a, b]))
    meas_spec = _MeasurementSpec(settings[0], {})

    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=settings,
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )

    should_be = [
        [4 / 3 * 10 ** 2, 0, 0],
        [0, 4 / 3 * 10 ** 2, 0],
        [0, 0, 4 / 3 * 10 ** 2],
    ]
    np.testing.assert_allclose(should_be, bsa.covariance())

    for setting in settings:
        np.testing.assert_allclose(0, bsa.mean(setting))
        np.testing.assert_allclose(100 / 4, bsa.variance(setting))
        np.testing.assert_allclose(10 / np.sqrt(4), bsa.stddev(setting))

    np.testing.assert_allclose([0, 0, 0], bsa.means())
