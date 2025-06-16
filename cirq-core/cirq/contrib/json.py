# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Functions for JSON serialization and de-serialization for classes in Contrib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirq.protocols.json_serialization import _register_resolver, DEFAULT_RESOLVERS

if TYPE_CHECKING:  # pragma: no cover
    from cirq.protocols.json_serialization import ObjectFactory

import functools


def contrib_class_resolver(cirq_type: str):
    """Extend cirq's JSON API with resolvers for cirq contrib classes."""
    return _class_resolver_dictionary().get(cirq_type, None)


@functools.lru_cache()
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


DEFAULT_CONTRIB_RESOLVERS = [contrib_class_resolver] + DEFAULT_RESOLVERS

_register_resolver(_class_resolver_dictionary)
