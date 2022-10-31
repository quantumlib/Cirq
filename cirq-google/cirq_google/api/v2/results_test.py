# pylint: disable=wrong-or-nonexistent-copyright-notice
import numpy as np
import pytest

import cirq
import cirq_google
from cirq_google.api import v2


@pytest.mark.parametrize('reps', range(1, 100, 7))
def test_pack_bits(reps):
    data = np.random.randint(2, size=reps, dtype=bool)
    packed = v2.pack_bits(data)
    assert isinstance(packed, bytes)
    assert len(packed) == (reps + 7) // 8
    unpacked = v2.unpack_bits(packed, reps)
    np.testing.assert_array_equal(unpacked, data)


q = cirq.GridQubit  # For brevity.


def _check_measurement(m, key, qubits, instances, invert_mask=None, tags=None):
    assert m.key == key
    assert m.qubits == qubits
    assert m.instances == instances
    if invert_mask is not None:
        assert m.invert_mask == invert_mask
    else:
        assert len(m.invert_mask) == len(m.qubits)
        assert m.invert_mask == [False] * len(m.qubits)
    if tags is not None:
        assert m.tags == tags
    else:
        assert len(m.tags) == 0


def test_find_measurements_simple_circuit():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k'))
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(m, 'k', [q(0, 0), q(0, 1), q(0, 2)], 1)


def test_find_measurements_invert_mask():
    circuit = cirq.Circuit()
    circuit.append(
        cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k', invert_mask=[False, True, True])
    )
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(m, 'k', [q(0, 0), q(0, 1), q(0, 2)], 1, [False, True, True])


def test_find_measurements_with_tags():
    circuit = cirq.Circuit()
    circuit.append(
        cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k', invert_mask=[False, True, True]).with_tags(
            cirq_google.CalibrationTag('special')
        )
    )
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(
        m,
        'k',
        [q(0, 0), q(0, 1), q(0, 2)],
        1,
        [False, True, True],
        [cirq_google.CalibrationTag('special')],
    )


def test_find_measurements_fill_mask():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k', invert_mask=[False, True]))
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(m, 'k', [q(0, 0), q(0, 1), q(0, 2)], 1, [False, True, False])


def test_find_measurements_repeated_keys():
    circuit = cirq.Circuit(
        cirq.measure(q(0, 0), q(0, 1), key='k'),
        cirq.measure(q(0, 1), q(0, 2), key='j'),
        cirq.measure(q(0, 0), q(0, 1), key='k'),
    )
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 2
    _check_measurement(measurements[0], 'k', [q(0, 0), q(0, 1)], 2)
    _check_measurement(measurements[1], 'j', [q(0, 1), q(0, 2)], 1)


def test_find_measurements_incompatible_repeated_keys():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), key='k'))
    circuit.append(cirq.measure(q(0, 1), q(0, 2), key='k'))
    with pytest.raises(ValueError, match='Incompatible repeated key'):
        v2.find_measurements(circuit)


def test_find_measurements_non_grid_qubits():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(cirq.NamedQubit('a'), key='k'))
    with pytest.raises(ValueError, match='Expected GridQubits'):
        v2.find_measurements(circuit)


def test_multiple_measurements_different_slots():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), key='k0'))
    circuit.append(cirq.measure(q(0, 2), q(0, 0), key='k1'))
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 2
    m0, m1 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 1)
    _check_measurement(m1, 'k1', [q(0, 2), q(0, 0)], 1)


def test_multiple_measurements_shared_slots():
    circuit = cirq.Circuit()
    circuit.append(
        [cirq.measure(q(0, 0), q(0, 1), key='k0'), cirq.measure(q(0, 2), q(1, 1), key='k1')]
    )
    circuit.append(
        [
            cirq.measure(q(1, 0), q(0, 0), q(0, 1), key='k2'),
            cirq.measure(q(1, 1), q(0, 2), key='k3'),
        ]
    )
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 4
    m0, m1, m2, m3 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 1)
    _check_measurement(m1, 'k1', [q(0, 2), q(1, 1)], 1)
    _check_measurement(m2, 'k2', [q(1, 0), q(0, 0), q(0, 1)], 1)
    _check_measurement(m3, 'k3', [q(1, 1), q(0, 2)], 1)


def test_results_to_proto():
    measurements = [v2.MeasureInfo('foo', [q(0, 0)], instances=1, invert_mask=[False], tags=[])]
    trial_results = [
        [
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 0}),
                records={'foo': np.array([[[0]], [[1]], [[0]], [[1]]], dtype=bool)},
            ),
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 1}),
                records={'foo': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool)},
            ),
        ],
        [
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 0}),
                records={'foo': np.array([[[0]], [[1]], [[0]], [[1]]], dtype=bool)},
            ),
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 1}),
                records={'foo': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool)},
            ),
        ],
    ]
    proto = v2.results_to_proto(trial_results, measurements)
    assert isinstance(proto, v2.result_pb2.Result)
    assert len(proto.sweep_results) == 2
    deserialized = v2.results_from_proto(proto, measurements)
    assert len(deserialized) == 2
    for sweep_results, expected in zip(deserialized, trial_results):
        assert len(sweep_results) == len(expected)
        for trial_result, expected_trial_result in zip(sweep_results, expected):
            assert trial_result.params == expected_trial_result.params
            assert trial_result.repetitions == expected_trial_result.repetitions
            np.testing.assert_array_equal(
                trial_result.measurements['foo'], expected_trial_result.measurements['foo']
            )


def test_results_to_proto_repeated_keys():
    measurements = [
        v2.MeasureInfo('foo', [q(0, 0)], instances=2, invert_mask=[False], tags=[]),
        v2.MeasureInfo('bar', [q(0, 0)], instances=1, invert_mask=[False], tags=[]),
    ]
    trial_results = [
        [
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 0}),
                records={
                    'foo': np.array([[[0], [0]], [[0], [1]], [[0], [0]], [[0], [1]]], dtype=bool),
                    'bar': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool),
                },
            ),
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 1}),
                records={
                    'foo': np.array([[[0], [0]], [[0], [1]], [[0], [1]], [[0], [0]]], dtype=bool),
                    'bar': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool),
                },
            ),
        ],
        [
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 0}),
                records={
                    'foo': np.array([[[0], [0]], [[0], [1]], [[0], [0]], [[0], [1]]], dtype=bool),
                    'bar': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool),
                },
            ),
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 1}),
                records={
                    'foo': np.array([[[0], [0]], [[0], [1]], [[0], [1]], [[0], [0]]], dtype=bool),
                    'bar': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool),
                },
            ),
        ],
    ]
    proto = v2.results_to_proto(trial_results, measurements)
    assert isinstance(proto, v2.result_pb2.Result)
    assert len(proto.sweep_results) == 2
    deserialized = v2.results_from_proto(proto, measurements)
    assert len(deserialized) == 2
    for sweep_results, expected in zip(deserialized, trial_results):
        assert len(sweep_results) == len(expected)
        for trial_result, expected_trial_result in zip(sweep_results, expected):
            assert trial_result.params == expected_trial_result.params
            assert trial_result.repetitions == expected_trial_result.repetitions
            for key in ['foo', 'bar']:
                np.testing.assert_array_equal(
                    trial_result.records[key], expected_trial_result.records[key]
                )


def test_results_to_proto_sweep_repetitions():
    measurements = [v2.MeasureInfo('foo', [q(0, 0)], instances=1, invert_mask=[False], tags=[])]
    trial_results = [
        [
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 0}), records={'foo': np.array([[[0]]], dtype=bool)}
            ),
            cirq.ResultDict(
                params=cirq.ParamResolver({'i': 1}),
                records={'foo': np.array([[[0]], [[1]]], dtype=bool)},
            ),
        ]
    ]
    with pytest.raises(ValueError, match='Different numbers of repetitions'):
        v2.results_to_proto(trial_results, measurements)


def test_results_from_proto_qubit_ordering():
    measurements = [
        v2.MeasureInfo(
            'foo',
            [q(0, 0), q(0, 1), q(1, 1)],
            instances=1,
            invert_mask=[False, False, False],
            tags=[],
        )
    ]
    proto = v2.result_pb2.Result()
    sr = proto.sweep_results.add()
    sr.repetitions = 8
    pr = sr.parameterized_results.add()
    pr.params.assignments.update({'i': 1})
    mr = pr.measurement_results.add()
    mr.key = 'foo'
    for qubit, results in [(q(0, 1), 0b1100_1100), (q(1, 1), 0b1010_1010), (q(0, 0), 0b1111_0000)]:
        qmr = mr.qubit_measurement_results.add()
        qmr.qubit.id = v2.qubit_to_proto_id(qubit)
        qmr.results = bytes([results])

    trial_results = v2.results_from_proto(proto, measurements)
    trial = trial_results[0][0]
    assert trial.params == cirq.ParamResolver({'i': 1})
    assert trial.repetitions == 8
    np.testing.assert_array_equal(
        trial.measurements['foo'],
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    )


def test_results_from_proto_repeated_keys():
    measurements = [
        v2.MeasureInfo(
            'foo',
            [q(0, 0), q(0, 1), q(1, 1)],
            instances=4,
            invert_mask=[False, False, False],
            tags=[],
        )
    ]
    proto = v2.result_pb2.Result()
    sr = proto.sweep_results.add()
    sr.repetitions = 8
    pr = sr.parameterized_results.add()
    pr.params.assignments.update({'i': 1})
    mr = pr.measurement_results.add()
    mr.key = 'foo'
    mr.instances = 4
    for qubit, results in [
        (q(0, 0), [0b1111_0000, 0b1101_0010, 0b1011_0100, 0b0111_1000]),
        (q(0, 1), [0b1100_1100, 0b1110_1110, 0b1000_1000, 0b0100_0100]),
        (q(1, 1), [0b1010_1010, 0b1000_1000, 0b1110_1110, 0b0010_0010]),
    ]:
        qmr = mr.qubit_measurement_results.add()
        qmr.qubit.id = v2.qubit_to_proto_id(qubit)
        qmr.results = bytes(results)

    trial_results = v2.results_from_proto(proto, measurements)
    trial = trial_results[0][0]
    assert trial.params == cirq.ParamResolver({'i': 1})
    assert trial.repetitions == 8
    np.testing.assert_array_equal(
        trial.records['foo'],
        np.array(
            [
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
                [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
                [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
                [[0, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
                [[1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]],
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                [[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0]],
            ],
            dtype=bool,
        ),
    )


def test_results_from_proto_duplicate_qubit():
    measurements = [
        v2.MeasureInfo(
            'foo',
            [q(0, 0), q(0, 1), q(1, 1)],
            instances=1,
            invert_mask=[False, False, False],
            tags=[],
        )
    ]
    proto = v2.result_pb2.Result()
    sr = proto.sweep_results.add()
    sr.repetitions = 8
    pr = sr.parameterized_results.add()
    pr.params.assignments.update({'i': 0})
    mr = pr.measurement_results.add()
    mr.key = 'foo'
    for qubit, results in [(q(0, 0), 0b1100_1100), (q(0, 1), 0b1010_1010), (q(0, 1), 0b1111_0000)]:
        qmr = mr.qubit_measurement_results.add()
        qmr.qubit.id = v2.qubit_to_proto_id(qubit)
        qmr.results = bytes([results])
    with pytest.raises(ValueError, match='Qubit already exists'):
        v2.results_from_proto(proto, measurements)


def test_results_from_proto_default_ordering():
    proto = v2.result_pb2.Result()
    sr = proto.sweep_results.add()
    sr.repetitions = 8
    pr = sr.parameterized_results.add()
    pr.params.assignments.update({'i': 1})
    mr = pr.measurement_results.add()
    mr.key = 'foo'
    for qubit, results in [(q(0, 1), 0b1100_1100), (q(1, 1), 0b1010_1010), (q(0, 0), 0b1111_0000)]:
        qmr = mr.qubit_measurement_results.add()
        qmr.qubit.id = v2.qubit_to_proto_id(qubit)
        qmr.results = bytes([results])

    trial_results = v2.results_from_proto(proto)
    trial = trial_results[0][0]
    assert trial.params == cirq.ParamResolver({'i': 1})
    assert trial.repetitions == 8
    np.testing.assert_array_equal(
        trial.measurements['foo'],
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    )
