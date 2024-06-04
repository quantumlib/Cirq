# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines which types are Sweepable."""

from typing import Iterable, Iterator, List, Sequence, Union, cast
import warnings
from typing_extensions import Protocol

from cirq._doc import document
from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import ListSweep, Points, Sweep, UnitSweep, Zip, dict_to_product_sweep

SweepLike = Union[ParamResolverOrSimilarType, Sweep]
document(SweepLike, """An object similar to an iterable of parameter resolvers.""")


class _Sweepable(Protocol):
    """An intermediate class allowing for recursive definition of Sweepable,
    since recursive union definitions are not yet supported in mypy."""

    def __iter__(self) -> Iterator[Union[SweepLike, '_Sweepable']]:
        pass


Sweepable = Union[SweepLike, _Sweepable]
document(Sweepable, """An object or collection of objects representing a parameter sweep.""")


def to_resolvers(sweepable: Sweepable) -> Iterator[ParamResolver]:
    """Convert a Sweepable to a list of ParamResolvers."""
    for sweep in to_sweeps(sweepable):
        yield from sweep


def to_sweeps(sweepable: Sweepable) -> List[Sweep]:
    """Converts a Sweepable to a list of Sweeps."""
    if sweepable is None:
        return [UnitSweep]
    if isinstance(sweepable, ParamResolver):
        return [_resolver_to_sweep(sweepable)]
    if isinstance(sweepable, Sweep):
        return [sweepable]
    if isinstance(sweepable, dict):
        if any(isinstance(val, Sequence) for val in sweepable.values()):
            warnings.warn(
                'Implicit expansion of a dictionary into a Cartesian product '
                'of sweeps is deprecated and will be removed in cirq 0.10. '
                'Instead, expand the sweep explicitly using '
                '`cirq.dict_to_product_sweep`.',
                DeprecationWarning,
                stacklevel=2,
            )
        product_sweep = dict_to_product_sweep(sweepable)
        return [_resolver_to_sweep(resolver) for resolver in product_sweep]
    if isinstance(sweepable, Iterable) and not isinstance(sweepable, str):
        return [sweep for item in sweepable for sweep in to_sweeps(item)]  # type: ignore[arg-type]
    raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\nsweepable: {sweepable}')


def to_sweep(
    sweep_or_resolver_list: Union[
        'Sweep', ParamResolverOrSimilarType, Iterable[ParamResolverOrSimilarType]
    ]
) -> 'Sweep':
    """Converts the argument into a ``cirq.Sweep``.

    Args:
        sweep_or_resolver_list: The object to try to turn into a
            ``cirq.Sweep`` . A ``cirq.Sweep``, a single ``cirq.ParamResolver``,
            or a list of ``cirq.ParamResolver`` s.

    Returns:
        A sweep equal to or containing the argument.

    Raises:
        TypeError: If an unsupport type was supplied.
    """
    if isinstance(sweep_or_resolver_list, Sweep):
        return sweep_or_resolver_list
    if isinstance(sweep_or_resolver_list, (ParamResolver, dict)):
        resolver = cast(ParamResolverOrSimilarType, sweep_or_resolver_list)
        return ListSweep([resolver])
    if isinstance(sweep_or_resolver_list, Iterable):
        resolver_iter = cast(Iterable[ParamResolverOrSimilarType], sweep_or_resolver_list)
        return ListSweep(resolver_iter)
    raise TypeError(f'Unexpected sweep-like value: {sweep_or_resolver_list}')


def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    params = resolver.param_dict
    if not params:
        return UnitSweep

    return Zip(*[_add_unit_to_metadata_if_present(key, value) for key, value in params.items()])


def _add_unit_to_metadata_if_present(k, v):
    """It is possible for the internal workflow to pass a value with units attached.
    We want to save the unit so it can be resolved later. We do this by saving the unit
    in DeviceParameter and attaching the object as metadata. This is a temporary solution
    until TUnits is implemented.
    """
    import cirq_google

    value_unit = str(v).split(" ")
    value = float(value_unit[0])
    unit = value_unit[1] if len(value_unit) > 1 else None

    if unit:
        # Append the unit as metadata so the parameter can be resolved server side during
        # internal use. See https://github.com/qh-lab/pyle/pull/46133.
        descriptor = cirq_google.study.DeviceParameter(path=None, value=value, units=unit)
        return Points(k, [cast(float, value)], metadata=descriptor)
    return Points(k, [cast(float, value)])
