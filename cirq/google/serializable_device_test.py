import pytest

import cirq
import cirq.google as cg


def test_foxtail():
    valid_qubit1 = cirq.GridQubit(0,0)
    valid_qubit2 = cirq.GridQubit(1,0)
    valid_qubit3 = cirq.GridQubit(1,1)
    invalid_qubit1 = cirq.GridQubit(2,2)
    invalid_qubit2 = cirq.GridQubit(2,3)

    foxtail = cg.SerializableDevice(
        proto = cg.known_devices.FOXTAIL_PROTO,
        gate_set=cg.gate_sets.XMON
    )
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

