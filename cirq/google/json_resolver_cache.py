from typing import Dict, Type, Optional


class _ResolverCache:
    """Lazily import and build registry to avoid circular imports."""

    def __init__(self):
        self._crd = None

    def __call__(self, cirq_type: str) -> Optional[Type]:
        return self.class_resolver_dictionary.get(cirq_type, None)

    @property
    def class_resolver_dictionary(self) -> Dict[str, Type]:
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


RESOLVER_CACHE = _ResolverCache()
