import pytest

import cirq
import cirq.google as cg


def test_foxtail():
    valid_qubit1 = cirq.GridQubit(0, 0)
    valid_qubit2 = cirq.GridQubit(1, 0)
    valid_qubit3 = cirq.GridQubit(1, 1)
    invalid_qubit1 = cirq.GridQubit(2, 2)
    invalid_qubit2 = cirq.GridQubit(2, 3)

    foxtail = cg.SerializableDevice(proto=cg.known_devices.FOXTAIL_PROTO,
                                    gate_set=cg.gate_sets.XMON)
    foxtail.validate_operation(cirq.X(valid_qubit1))
    foxtail.validate_operation(cirq.X(valid_qubit2))
    foxtail.validate_operation(cirq.X(valid_qubit3))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(invalid_qubit1))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(invalid_qubit2))
    foxtail.validate_operation(cirq.CZ(valid_qubit1, valid_qubit2))
    foxtail.validate_operation(cirq.CZ(valid_qubit2, valid_qubit1))
    # Non-local
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.CZ(valid_qubit1, valid_qubit3))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.CZ(invalid_qubit1, invalid_qubit2))

    # Unsupport op
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.H(invalid_qubit1))


def test_duration_of():
    # TODO(dstrain): Finish implementing duration_of calls
    valid_qubit1 = cirq.GridQubit(0, 0)

    foxtail = cg.SerializableDevice(proto=cg.known_devices.FOXTAIL_PROTO,
                                    gate_set=cg.gate_sets.XMON)

    assert foxtail.duration_of(cirq.X(valid_qubit1)) == cirq.Duration(nanos=20)

    # Unsupport op
    with pytest.raises(ValueError):
        assert foxtail.duration_of(cirq.H(valid_qubit1))
