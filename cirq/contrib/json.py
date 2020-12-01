"""Contains functions for adding JSON serialization and de-serialization for
classes in Contrib.

"""
from typing import Dict

from cirq.protocols.json_serialization import _internal_register_resolver, ObjectFactory


def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:
    """Extend cirq's JSON API with resolvers for cirq contrib classes."""
    from cirq.contrib.quantum_volume import QuantumVolumeResult
    from cirq.contrib.acquaintance import SwapPermutationGate

    classes = [
        QuantumVolumeResult,
        SwapPermutationGate,
    ]
    return {cls.__name__: cls for cls in classes}


# TODO(https://github.com/quantumlib/Cirq/issues/3555): contrib should be
#  properly tested
_internal_register_resolver(_class_resolver_dictionary)
