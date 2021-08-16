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
import tempfile

import numpy as np
import pytest

import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator
from cirq.work.observable_measurement import (
    _with_parameterized_layers,
    _get_params_for_setting,
    _pad_setting,
    _subdivide_meas_specs,
    _aggregate_n_repetitions,
    _check_meas_specs_still_todo,
    StoppingCriteria,
    _parse_checkpoint_options,
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


def test_aggregate_n_repetitions():
    with pytest.warns(UserWarning):
        reps = _aggregate_n_repetitions({5, 6})
    assert reps == 6


class _MockBitstringAccumulator(BitstringAccumulator):
    def __init__(self):
        super().__init__(
            meas_spec=None,
            simul_settings=None,
            qubit_to_index={q: i for i, q in enumerate(cirq.LineQubit.range(5))},
        )

    def covariance(self, *, atol=1e-8) -> np.ndarray:
        cov = np.cov(self.bitstrings.T, ddof=1)
        assert cov.shape == (5, 5)
        return cov / len(self.bitstrings)


def test_variance_stopping_criteria():
    stop = cw.VarianceStoppingCriteria(variance_bound=1e-6)
    acc = _MockBitstringAccumulator()
    assert stop.more_repetitions(acc) == 10_000
    rs = np.random.RandomState(52)

    # small number of results
    acc.consume_results(rs.choice([0, 1], size=(100, 5)).astype(np.uint8))
    assert stop.more_repetitions(acc) == 10_000

    # large number of results
    acc.consume_results(rs.choice([0, 1], size=(10_000, 5)).astype(np.uint8))
    assert stop.more_repetitions(acc) == 0


class _WildVarianceStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self._state = 0

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        """Ignore everything, request either 5 or 6 repetitions."""
        self._state += 1
        return [5, 6][self._state % 2]


def test_variance_stopping_criteria_aggregate_n_repetitions():
    stop = _WildVarianceStoppingCriteria()
    acc1 = _MockBitstringAccumulator()
    acc2 = _MockBitstringAccumulator()
    accumulators = {'FakeMeasSpec1': acc1, 'FakeMeasSpec2': acc2}
    with pytest.warns(UserWarning, match='the largest value will be used: 6.'):
        still_todo, reps = _check_meas_specs_still_todo(
            meas_specs=sorted(accumulators.keys()),
            accumulators=accumulators,
            stopping_criteria=stop,
        )
    assert still_todo == ['FakeMeasSpec1', 'FakeMeasSpec2']
    assert reps == 6


def test_repetitions_stopping_criteria():
    stop = cw.RepetitionsStoppingCriteria(total_repetitions=50_000)
    acc = _MockBitstringAccumulator()

    todos = [stop.more_repetitions(acc)]
    for _ in range(6):
        acc.consume_results(np.zeros((10_000, 5), dtype=np.uint8))
        todos.append(stop.more_repetitions(acc))
    assert todos == [10_000] * 5 + [0, 0]


def test_repetitions_stopping_criteria_partial():
    stop = cw.RepetitionsStoppingCriteria(total_repetitions=5_000, repetitions_per_chunk=1_000_000)
    acc = _MockBitstringAccumulator()
    assert stop.more_repetitions(acc) == 5_000


def _set_up_meas_specs_for_testing():
    q0, q1 = cirq.LineQubit.range(2)
    setting = cw.InitObsSetting(
        init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1)
    )
    meas_spec = _MeasurementSpec(
        max_setting=setting,
        circuit_params={
            'beta': 0.123,
            'gamma': 0.456,
        },
    )
    bsa = cw.BitstringAccumulator(
        meas_spec, [], {q: i for i, q in enumerate(cirq.LineQubit.range(3))}
    )
    return bsa, meas_spec


def test_meas_specs_still_todo():
    bsa, meas_spec = _set_up_meas_specs_for_testing()
    stop = cw.RepetitionsStoppingCriteria(1_000)

    # 1. before taking any data
    still_todo, reps = _check_meas_specs_still_todo(
        meas_specs=[meas_spec],
        accumulators={meas_spec: bsa},
        stopping_criteria=stop,
    )
    assert still_todo == [meas_spec]
    assert reps == 1_000

    # 2. After taking a mocked-out 997 shots.
    bsa.consume_results(np.zeros((997, 3), dtype=np.uint8))
    still_todo, reps = _check_meas_specs_still_todo(
        meas_specs=[meas_spec],
        accumulators={meas_spec: bsa},
        stopping_criteria=stop,
    )
    assert still_todo == [meas_spec]
    assert reps == 3

    # 3. After taking the final 3 shots
    bsa.consume_results(np.zeros((reps, 3), dtype=np.uint8))
    still_todo, reps = _check_meas_specs_still_todo(
        meas_specs=[meas_spec],
        accumulators={meas_spec: bsa},
        stopping_criteria=stop,
    )
    assert still_todo == []
    assert reps == 0


def test_meas_spec_still_todo_bad_spec():
    bsa, meas_spec = _set_up_meas_specs_for_testing()

    class BadStopping(StoppingCriteria):
        def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
            return -23

    bad_stop = BadStopping()
    with pytest.raises(ValueError, match='positive'):
        _, _ = _check_meas_specs_still_todo(
            meas_specs=[meas_spec],
            accumulators={meas_spec: bsa},
            stopping_criteria=bad_stop,
        )


def test_meas_spec_still_todo_too_many_params(monkeypatch):
    monkeypatch.setattr(cw.observable_measurement, 'MAX_REPETITIONS_PER_JOB', 30_000)
    bsa, meas_spec = _set_up_meas_specs_for_testing()
    lots_of_meas_spec = [meas_spec] * 3_001
    stop = cw.RepetitionsStoppingCriteria(10_000)
    with pytest.raises(ValueError, match='too many parameter settings'):
        _, _ = _check_meas_specs_still_todo(
            meas_specs=lots_of_meas_spec,
            accumulators={meas_spec: bsa},
            stopping_criteria=stop,
        )


def test_meas_spec_still_todo_lots_of_params(monkeypatch):
    monkeypatch.setattr(cw.observable_measurement, 'MAX_REPETITIONS_PER_JOB', 30_000)
    bsa, meas_spec = _set_up_meas_specs_for_testing()
    lots_of_meas_spec = [meas_spec] * 4
    stop = cw.RepetitionsStoppingCriteria(10_000)
    with pytest.warns(UserWarning, match='will be throttled from 10000 to 7500'):
        _, _ = _check_meas_specs_still_todo(
            meas_specs=lots_of_meas_spec,
            accumulators={meas_spec: bsa},
            stopping_criteria=stop,
        )


def test_checkpoint_options():
    # There are three ~binary options (the latter two can be either specified or `None`. We
    # test those 2^3 cases.

    assert _parse_checkpoint_options(False, None, None) == (None, None)
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, 'test', None)
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, None, 'test')
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, 'test1', 'test2')

    chk, chkprev = _parse_checkpoint_options(True, None, None)
    assert chk.startswith(tempfile.gettempdir())
    assert chk.endswith('observables.json')
    assert chkprev.startswith(tempfile.gettempdir())
    assert chkprev.endswith('observables.prev.json')

    chk, chkprev = _parse_checkpoint_options(True, None, 'prev.json')
    assert chk.startswith(tempfile.gettempdir())
    assert chk.endswith('observables.json')
    assert chkprev == 'prev.json'

    chk, chkprev = _parse_checkpoint_options(True, 'my_fancy_observables.json', None)
    assert chk == 'my_fancy_observables.json'
    assert chkprev == 'my_fancy_observables.prev.json'

    chk, chkprev = _parse_checkpoint_options(True, 'my_fancy/observables.json', None)
    assert chk == 'my_fancy/observables.json'
    assert chkprev == 'my_fancy/observables.prev.json'

    with pytest.raises(ValueError, match=r'Please use a `.json` filename.*'):
        _parse_checkpoint_options(True, 'my_fancy_observables.obs', None)

    with pytest.raises(ValueError, match=r"pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, 'my_fancy_observables', None)
    with pytest.raises(ValueError, match=r"pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, '.obs', None)
    with pytest.raises(ValueError, match=r"pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, 'obs.', None)
    with pytest.raises(ValueError, match=r"pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, '', None)

    chk, chkprev = _parse_checkpoint_options(True, 'test1', 'test2')
    assert chk == 'test1'
    assert chkprev == 'test2'


@pytest.mark.parametrize(('with_circuit_sweep', 'checkpoint'), [(True, True), (False, False)])
def test_measure_grouped_settings(with_circuit_sweep, checkpoint, tmpdir):
    qubits = cirq.LineQubit.range(1)
    (q,) = qubits
    tests = [
        (cirq.KET_ZERO, cirq.Z, 1),
        (cirq.KET_ONE, cirq.Z, -1),
        (cirq.KET_PLUS, cirq.X, 1),
        (cirq.KET_MINUS, cirq.X, -1),
        (cirq.KET_IMAG, cirq.Y, 1),
        (cirq.KET_MINUS_IMAG, cirq.Y, -1),
    ]
    if with_circuit_sweep:
        ss = cirq.Linspace('a', 0, 1, 12)
    else:
        ss = None

    if checkpoint:
        checkpoint_fn = f'{tmpdir}/obs.json'
    else:
        checkpoint_fn = None

    for init, obs, coef in tests:
        setting = cw.InitObsSetting(
            init_state=init(q),
            observable=obs(q),
        )
        grouped_settings = {setting: [setting]}
        circuit = cirq.Circuit(cirq.I.on_each(*qubits))
        results = cw.measure_grouped_settings(
            circuit=circuit,
            grouped_settings=grouped_settings,
            sampler=cirq.Simulator(),
            stopping_criteria=cw.RepetitionsStoppingCriteria(1_000, repetitions_per_chunk=500),
            circuit_sweep=ss,
            checkpoint=checkpoint,
            checkpoint_fn=checkpoint_fn,
        )
        if with_circuit_sweep:
            for result in results:
                assert result.means() == [coef]
        else:
            (result,) = results  # one group
            assert result.means() == [coef]


def _get_some_grouped_settings():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    terms = [
        cirq.X(q0),
        cirq.Y(q1),
    ]
    settings = list(cirq.work.observables_to_settings(terms, qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    return grouped_settings, qubits


def test_measure_grouped_settings_calibration_validation():
    dummy_ro_calib = _MockBitstringAccumulator()
    grouped_settings, qubits = _get_some_grouped_settings()

    with pytest.raises(
        ValueError, match=r'Readout calibration only works if `readout_symmetrization` is enabled'
    ):
        cw.measure_grouped_settings(
            circuit=cirq.Circuit(cirq.I.on_each(*qubits)),
            grouped_settings=grouped_settings,
            sampler=cirq.Simulator(),
            stopping_criteria=cw.RepetitionsStoppingCriteria(10_000),
            readout_calibrations=dummy_ro_calib,
            readout_symmetrization=False,  # no-no!
        )


def test_measure_grouped_settings_read_checkpoint(tmpdir):
    qubits = cirq.LineQubit.range(1)
    (q,) = qubits

    setting = cw.InitObsSetting(
        init_state=cirq.KET_ZERO(q),
        observable=cirq.Z(q),
    )
    grouped_settings = {setting: [setting]}
    circuit = cirq.Circuit(cirq.I.on_each(*qubits))
    with pytest.raises(ValueError, match=r'same filename.*'):
        _ = cw.measure_grouped_settings(
            circuit=circuit,
            grouped_settings=grouped_settings,
            sampler=cirq.Simulator(),
            stopping_criteria=cw.RepetitionsStoppingCriteria(1_000, repetitions_per_chunk=500),
            checkpoint=True,
            checkpoint_fn=f'{tmpdir}/obs.json',
            checkpoint_other_fn=f'{tmpdir}/obs.json',  # Same filename
        )
    _ = cw.measure_grouped_settings(
        circuit=circuit,
        grouped_settings=grouped_settings,
        sampler=cirq.Simulator(),
        stopping_criteria=cw.RepetitionsStoppingCriteria(1_000, repetitions_per_chunk=500),
        checkpoint=True,
        checkpoint_fn=f'{tmpdir}/obs.json',
        checkpoint_other_fn=f'{tmpdir}/obs.prev.json',
    )
    results = cirq.read_json(f'{tmpdir}/obs.json')
    (result,) = results  # one group
    assert result.n_repetitions == 1_000
    assert result.means() == [1.0]
