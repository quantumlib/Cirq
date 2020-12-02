import inspect
import pathlib
from typing import Set, Dict, Tuple, Any, List

import cirq


def _all_public() -> Set[str]:
    module_scopes = [
        (cirq, 'cirq'),
        (cirq.experiments, 'cirq.experiments'),
        (cirq.google, 'cirq.google'),
        (cirq.testing, 'cirq.testing'),
    ]

    by_name: Dict[str, Tuple[str, Any]] = {}
    result: Set[str] = set()
    for module, scope in module_scopes:
        for name, obj in inspect.getmembers(module):
            if name.startswith('_') or inspect.ismodule(obj):
                continue

            full_name = f'{scope}.{name}'
            if name in by_name:
                old_full_name, old_obj = by_name[name]
                if obj is not old_obj:
                    # coverage: ignore
                    raise ValueError(f'Ambiguous name:\n{old_full_name}\n{full_name}\n')
                if len(full_name) > len(old_full_name):
                    continue

                # coverage: ignore
                result.remove(old_full_name)

            by_name[name] = (full_name, obj)
            result.add(full_name)
    return result


def _api_rst_fullnames_per_section() -> List[List[str]]:
    result: List[List[str]] = []
    section: List[str] = []
    seen: Set[str] = set()
    with open(pathlib.Path(__file__).parent / 'api.rst', mode='r') as f:
        for line in f.readlines():
            if line.strip() == '.. autosummary::':
                if section:
                    result.append(section)
                    section = []
            elif (
                '    cirq.' in line
                or '    .. autoclass:: cirq.' in line
                or '    .. autofunction:: cirq.' in line
            ):
                fullname = line[line.find('cirq') :].strip()
                if fullname in seen:
                    # coverage: ignore
                    raise ValueError(f'{fullname} appears twice in api.rst')
                section.append(fullname)
                seen.add(fullname)
        if section:
            result.append(section)
    return result


def test_public_values_equals_documented_values():
    in_actual_api = _all_public()
    in_api_reference = {
        fullname for section in _api_rst_fullnames_per_section() for fullname in section
    }
    unlisted = in_actual_api - in_api_reference
    hidden = {
        fullname
        for fullname in in_api_reference - in_actual_api
        if not fullname.startswith('cirq.contrib.')
    }
    assert (
        not unlisted
    ), 'Public class/method/value not listed in rtd_docs/api.rst:\n    ' + '\n    '.join(
        sorted(unlisted)
    )
    assert not hidden, (
        'Private or non-existent class/method/value listed in rtd_docs/api.rst:'
        '\n    ' + '\n    '.join(sorted(hidden))
    )


def test_api_rst_sorted():
    def order(fullname: str) -> Any:
        name = fullname.split('.')[-1]
        start = fullname[: -len(name)]
        return (
            # First sort by package.
            start,
            # Then by tiny-ness (len=1 then len=2 then other).
            min(len(name), 3),
            # Then by type (constants then methods then classes).
            name != name.upper(),
            name != name.lower(),
            # Then by name.
            fullname,
        )

    for section in _api_rst_fullnames_per_section():
        sorted_section = sorted(section, key=order)
        assert (
            section == sorted_section
        ), f"A section in api.rst is not sorted. Should be:\n    " + '\n    '.join(sorted_section)
