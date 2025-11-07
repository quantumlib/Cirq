# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Functions for JSON serialization and de-serialization for classes in Contrib."""

from __future__ import annotations
import warnings
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

class _DeprecatedDefaultContribResolvers(list):
    """Wrapper to emit deprecation warning when DEFAULT_CONTRIB_RESOLVERS is accessed."""

    def __init__(self):
        super().__init__([contrib_class_resolver] + DEFAULT_RESOLVERS)
        self._warning_shown = False

    def __iter__(self):
        self._show_warning()
        return super().__iter__()

    def __getitem__(self, item):
        self._show_warning()
        return super().__getitem__(item)

    def _show_warning(self):
        if not self._warning_shown:
            warnings.warn(
                "DEFAULT_CONTRIB_RESOLVERS is deprecated. "
                "Contrib classes are now automatically resolved through the standard JSON resolver. "
                "You can remove the 'resolvers' parameter from assert_json_roundtrip_works calls.",
                DeprecationWarning,
                stacklevel=4
            )
            self._warning_shown = True


DEFAULT_CONTRIB_RESOLVERS = _DeprecatedDefaultContribResolvers()


_register_resolver(_class_resolver_dictionary)
