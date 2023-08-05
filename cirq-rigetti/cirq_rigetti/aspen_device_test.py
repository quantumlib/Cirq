# pylint: disable=wrong-or-nonexistent-copyright-notice
import os
from unittest.mock import patch, PropertyMock
from math import sqrt
import pathlib
import json
import pytest
import cirq
from cirq_rigetti import (
    OctagonalQubit,
    AspenQubit,
    RigettiQCSAspenDevice,
    UnsupportedQubit,
    UnsupportedRigettiQCSOperation,
    UnsupportedRigettiQCSQuantumProcessor,
)
from qcs_api_client.models import InstructionSetArchitecture, Node
import numpy as np

dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
fixture_path = dir_path / '__fixtures__'


@pytest.fixture
def qcs_aspen8_isa() -> InstructionSetArchitecture:
    with open(fixture_path / 'QCS-Aspen-8-ISA.json', 'r') as f:
        return InstructionSetArchitecture.from_dict(json.load(f))


def test_octagonal_qubit_index():
    """test that OctagonalQubit properly calculates index and uses it for comparison"""
    qubit0 = OctagonalQubit(0)
    assert qubit0.index == 0
    assert OctagonalQubit(1) > qubit0


def test_octagonal_qubit_repr():
    """test OctagonalQubit.__repr__"""
    qubit5 = OctagonalQubit(5)
    assert "cirq_rigetti.OctagonalQubit(octagon_position=5)" == repr(qubit5)


def test_octagonal_qubit_positions():
    """test OctagonalQubit 2D position and distance calculations"""
    qubit0 = OctagonalQubit(0)
    assert qubit0.octagon_position == 0
    assert qubit0.dimension == 2

    qubit5 = OctagonalQubit(5)
    assert qubit5.x == 0
    assert np.isclose(qubit5.y, 1 / sqrt(2))
    assert qubit5.z == 0

    qubit3 = OctagonalQubit(3)
    assert np.isclose(qubit3.x, 1 + 1 / sqrt(2))
    assert qubit3.y == 0
    assert qubit3.z == 0

    qubit2 = OctagonalQubit(2)
    assert np.isclose(qubit2.x, 1 + sqrt(2))
    assert np.isclose(qubit2.y, 1 / sqrt(2))
    assert qubit2.z == 0

    with patch(
        'cirq_rigetti.OctagonalQubit.octagon_position', new_callable=PropertyMock
    ) as mock_octagon_position:
        mock_octagon_position.return_value = 9
        invalid_qubit = OctagonalQubit(0)
        with pytest.raises(ValueError):
            _ = invalid_qubit.x
        with pytest.raises(ValueError):
            _ = invalid_qubit.y

    qubit0 = OctagonalQubit(0)
    assert np.isclose(qubit0.distance(OctagonalQubit(1)), 1)
    assert qubit0.distance(OctagonalQubit(7)) == 1

    with pytest.raises(TypeError):
        _ = qubit0.distance(AspenQubit(0, 0))


def test_octagonal_position_validation():
    """test OctagonalQubit validates octagon position when initialized"""
    with pytest.raises(ValueError):
        _ = OctagonalQubit(8)


def test_aspen_qubit_index():
    """test that AspenQubit properly calculates index and uses it for comparison"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.index == 10
    assert qubit10 > AspenQubit(0, 5)


def test_aspen_qubit_repr():
    """test AspenQubit.__repr__"""
    qubit10 = AspenQubit(1, 0)
    assert "cirq_rigetti.AspenQubit(octagon=1, octagon_position=0)" == repr(qubit10)


def test_aspen_qubit_positions_and_distance():
    """test AspenQubit 2D position and distance calculations"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.octagon == 1
    assert qubit10.octagon_position == 0
    assert qubit10.dimension == 2

    assert np.isclose(qubit10.x, 3 + 3 / sqrt(2))
    assert np.isclose(qubit10.y, 1 + sqrt(2))
    assert np.isclose(qubit10.distance(AspenQubit(0, 7)), 3 + 2 / sqrt(2))
    assert np.isclose(qubit10.distance(AspenQubit(1, 3)), 1 + sqrt(2))

    qubit15 = AspenQubit(1, 5)
    assert np.isclose(qubit15.x, (2 + sqrt(2)))

    qubit15 = AspenQubit(1, 1)
    assert np.isclose(qubit15.x, (2 + sqrt(2)) + 1 + sqrt(2))

    with patch(
        'cirq_rigetti.AspenQubit.octagon_position', new_callable=PropertyMock
    ) as mock_octagon_position:
        mock_octagon_position.return_value = 9
        invalid_qubit = AspenQubit(0, 0)
        with pytest.raises(ValueError):
            _ = invalid_qubit.x
        with pytest.raises(ValueError):
            _ = invalid_qubit.y

    with pytest.raises(TypeError):
        _ = qubit10.distance(OctagonalQubit(0))

    with pytest.raises(ValueError):
        _ = AspenQubit(1, 9)


def test_aspen_qubit_qid_conversions():
    """test AspenQubit conversion to and from other `cirq.Qid` implementations"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.to_named_qubit() == cirq.NamedQubit('10')
    assert AspenQubit.from_named_qubit(cirq.NamedQubit('10')) == AspenQubit(1, 0)
    with pytest.raises(ValueError):
        _ = AspenQubit.from_named_qubit(cirq.NamedQubit('s'))
    with pytest.raises(ValueError):
        _ = AspenQubit.from_named_qubit(cirq.NamedQubit('19'))

    with pytest.raises(ValueError):
        _ = qubit10.to_grid_qubit()
    assert AspenQubit(0, 2).to_grid_qubit() == cirq.GridQubit(0, 0)
    assert AspenQubit(0, 1).to_grid_qubit() == cirq.GridQubit(1, 0)
    assert AspenQubit(1, 5).to_grid_qubit() == cirq.GridQubit(0, 1)
    assert AspenQubit(1, 6).to_grid_qubit() == cirq.GridQubit(1, 1)
    assert AspenQubit.from_grid_qubit(cirq.GridQubit(1, 1)) == AspenQubit(1, 6)
    with pytest.raises(ValueError):
        _ = AspenQubit.from_grid_qubit(cirq.GridQubit(3, 4))


def test_rigetti_qcs_aspen_device_topology(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice topological nodes and edges"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)

    assert len(device.qubit_topology.nodes) == 32
    assert len(device.qubit_topology.edges) == 38


@pytest.mark.parametrize(
    'qubit',
    [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(1, 1),
        cirq.LineQubit(30),
        cirq.NamedQubit('33'),
        AspenQubit(3, 6),
        OctagonalQubit(6),
    ],
)
def test_rigetti_qcs_aspen_device_valid_qubit(
    qubit: cirq.Qid, qcs_aspen8_isa: InstructionSetArchitecture
):
    """test RigettiQCSAspenDevice throws no error on valid qubits or operations on those qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    device.validate_qubit(qubit)
    device.validate_operation(cirq.I(qubit))


@pytest.mark.parametrize(
    'qubit',
    [
        cirq.GridQubit(2, 2),
        cirq.LineQubit(33),
        cirq.NamedQubit('s'),
        cirq.NamedQubit('40'),
        cirq.NamedQubit('9'),
        AspenQubit(4, 0),
    ],
)
def test_rigetti_qcs_aspen_device_invalid_qubit(
    qubit: cirq.Qid, qcs_aspen8_isa: InstructionSetArchitecture
):
    """test RigettiQCSAspenDevice throws error on invalid qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    with pytest.raises(UnsupportedQubit):
        device.validate_qubit(qubit)
    with pytest.raises(UnsupportedQubit):
        device.validate_operation(cirq.I(qubit))


def test_rigetti_qcs_aspen_device_non_existent_qubit(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice throws error when qubit does not exist on device"""
    # test device may only be initialized with Aspen ISA.
    device_with_limited_nodes = RigettiQCSAspenDevice(
        isa=InstructionSetArchitecture.from_dict(qcs_aspen8_isa.to_dict())
    )
    device_with_limited_nodes.isa.architecture.nodes = [Node(node_id=10)]
    with pytest.raises(UnsupportedQubit):
        device_with_limited_nodes.validate_qubit(cirq.GridQubit(0, 0))


@pytest.mark.parametrize(
    'operation',
    [
        cirq.CNOT(OctagonalQubit(0), OctagonalQubit(2)),
        cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
        cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)),
        cirq.CNOT(cirq.NamedQubit('0'), cirq.NamedQubit('2')),
        cirq.CNOT(AspenQubit(0, 1), AspenQubit(1, 1)),
    ],
)
def test_rigetti_qcs_aspen_device_invalid_operation(
    operation: cirq.Operation, qcs_aspen8_isa: InstructionSetArchitecture
):
    """test RigettiQCSAspenDevice throws error when validating 2Q operations on
    non-adjacent qubits
    """
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    with pytest.raises(UnsupportedRigettiQCSOperation):
        device.validate_operation(operation)


@pytest.mark.parametrize('operation', [cirq.CNOT(AspenQubit(0, 1), AspenQubit(0, 2))])
def test_rigetti_qcs_aspen_device_valid_operation(
    operation: cirq.Operation, qcs_aspen8_isa: InstructionSetArchitecture
):
    """test RigettiQCSAspenDevice throws no error when validating 2Q operations on
    adjacent qubits
    """
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    device.validate_operation(operation)


def test_rigetti_qcs_aspen_device_qubits(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice returns accurate set of qubits"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    expected_qubits = set()
    for i in range(4):
        for j in range(8):
            expected_qubits.add(AspenQubit(octagon=i, octagon_position=j))
    assert expected_qubits == set(device.qubits())


def test_rigetti_qcs_aspen_device_repr(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice.__repr__"""
    device = RigettiQCSAspenDevice(isa=qcs_aspen8_isa)
    assert f'cirq_rigetti.RigettiQCSAspenDevice(isa={qcs_aspen8_isa!r})' == repr(device)


def test_rigetti_qcs_aspen_device_family_validation(qcs_aspen8_isa: InstructionSetArchitecture):
    """test RigettiQCSAspenDevice validates architecture family on initialization"""
    non_aspen_isa = InstructionSetArchitecture.from_dict(qcs_aspen8_isa.to_dict())
    non_aspen_isa.architecture.family = "not-aspen"  # type: ignore
    with pytest.raises(UnsupportedRigettiQCSQuantumProcessor):
        RigettiQCSAspenDevice(isa=non_aspen_isa)
