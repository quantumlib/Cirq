"""Contains functions for adding JSON serialization and de-serialization for
classes in Contrib.

"""

from cirq.protocols.json_serialization import DEFAULT_RESOLVERS


def contrib_class_resolver(cirq_type: str):
    """Extend cirq's JSON API with resolvers for cirq contrib classes."""
    from cirq.contrib.quantum_volume import QuantumVolumeResult
    from cirq.contrib.acquaintance import SwapPermutationGate
    classes = [
        QuantumVolumeResult,
        SwapPermutationGate,
    ]
    d = {cls.__name__: cls for cls in classes}
    return d.get(cirq_type, None)


DEFAULT_CONTRIB_RESOLVERS = [contrib_class_resolver] + DEFAULT_RESOLVERS
