import functools
from typing import Dict

from cirq.protocols.json_serialization import ObjectFactory, lazy_resolver


@functools.lru_cache(maxsize=1)
def _class_resolver_dictionary(self) -> Dict[str, ObjectFactory]:
    if self._crd is not None:
        return self._crd

    import cirq.google
    self._crd = {
        'SycamoreGate': cirq.google.SycamoreGate,
        'GateTabulation': cirq.google.GateTabulation,
        'PhysicalZTag': cirq.google.PhysicalZTag,
        'CalibrationTag': cirq.google.CalibrationTag,
    }
    return self._crd


def _register_json_resolver():
    from cirq.protocols.json_serialization import DEFAULT_RESOLVERS
    DEFAULT_RESOLVERS.append(lazy_resolver(_class_resolver_dictionary))


_register_json_resolver()
