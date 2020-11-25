import functools
from typing import Dict

from cirq.protocols.json_serialization import ObjectFactory


@functools.lru_cache(maxsize=1)
def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:
    import cirq.google
    return {
        'Calibration': cirq.google.Calibration,
        'CalibrationTag': cirq.google.CalibrationTag,
        'SycamoreGate': cirq.google.SycamoreGate,
        'GateTabulation': cirq.google.GateTabulation,
        'PhysicalZTag': cirq.google.PhysicalZTag,
    }
