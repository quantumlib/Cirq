import importlib
import pickle
from unittest import mock

import applications_superstaq
import cirq
import pytest

from cirq_superstaq import aqt


def test_aqt_out_repr() -> None:
    circuit = cirq.Circuit()
    assert repr(aqt.AQTCompilerOutput(circuit)) == f"AQTCompilerOutput({circuit!r}, None, None)"

    circuits = [circuit, circuit]
    assert repr(aqt.AQTCompilerOutput(circuits)) == f"AQTCompilerOutput({circuits!r}, None, None)"


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(aqt)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = applications_superstaq.converters.serialize({})
    pulse_lists_str = applications_superstaq.converters.serialize([[[]]])

    json_dict: dict

    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit)],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = aqt.read_json(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    # multiple circuits
    pulse_lists_str = applications_superstaq.converters.serialize([[[]], [[]]])
    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit), cirq.to_json(circuit)],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = applications_superstaq.converters.serialize(seq.__getstate__())
    pulse_lists_str = applications_superstaq.converters.serialize([[[]]])

    json_dict: dict

    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit)],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = aqt.read_json(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_list == [[]]
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    # multiple circuits
    pulse_lists_str = applications_superstaq.converters.serialize([[[]], [[]]])
    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit), cirq.to_json(circuit)],
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = aqt.read_json(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert out.pulse_lists == [[[]], [[]]]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
