# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Functions for JSON serialization and de-serialization for classes in Contrib."""

from __future__ import annotations
import functools

from cirq.protocols.json_serialization import _register_resolver, DEFAULT_RESOLVERS, ObjectFactory


def contrib_class_resolver(cirq_type: str) -> ObjectFactory | None:
    """Extend cirq's JSON API with resolvers for cirq contrib classes."""
    return _class_resolver_dictionary().get(cirq_type, None)


@functools.cache
def _class_resolver_dictionary() -> dict[str, ObjectFactory]:
    from cirq.contrib.acquaintance import SwapPermutationGate
    from cirq.contrib.bayesian_network import BayesianNetworkGate
    from cirq.contrib.noise_models import (
        DampedReadoutNoiseModel,
        DepolarizingNoiseModel,
        DepolarizingWithDampedReadoutNoiseModel,
        DepolarizingWithReadoutNoiseModel,
        ReadoutNoiseModel,
    )
    from cirq.contrib.quantum_volume import QuantumVolumeResult

    classes = [
        BayesianNetworkGate,
        QuantumVolumeResult,
        SwapPermutationGate,
        DepolarizingNoiseModel,
        ReadoutNoiseModel,
        DampedReadoutNoiseModel,
        DepolarizingWithReadoutNoiseModel,
        DepolarizingWithDampedReadoutNoiseModel,
    ]
    return {cls.__name__: cls for cls in classes}

_register_resolver(_class_resolver_dictionary)

DEFAULT_CONTRIB_RESOLVERS = [contrib_class_resolver, *DEFAULT_RESOLVERS]

_DEFAULT_CONTRIB_RESOLVERS_DEPRECATION_MESSAGE = (
    'DEFAULT_CONTRIB_RESOLVERS will no longer be supported.'
    'Contrib classes are now automatically resolved through the standard JSON resolver.'
    'You can remove the "resolvers" parameter from assert_json_roundtrip_works calls.'
)
from cirq import _compat

_compat.deprecate_attributes(
    __name__,
    {
        'DEFAULT_CONTRIB_RESOLVERS': ('v1.8', _DEFAULT_CONTRIB_RESOLVERS_DEPRECATION_MESSAGE),
    },
)