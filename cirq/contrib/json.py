"""Contains functions for adding JSON serialization and de-serialization for
classes in Contrib.

"""


def contrib_class_resolver(cirq_type: str):
    """Extend cirq's JSON API with resolvers for cirq contrib classes."""
    from cirq.contrib.quantum_volume import QuantumVolumeResult
    classes = [
        QuantumVolumeResult,
    ]
    print(QuantumVolumeResult.__name__)
    d = {cls.__name__: cls for cls in classes}
    return d.get(cirq_type, None)
