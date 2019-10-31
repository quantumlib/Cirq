from collections import defaultdict
from typing import NamedTuple

from docs.docs_coverage_test import _find_public_cirq_members


KnownCategory = NamedTuple(
    'KnownCategory',
    [
        ('id', str),
        ('title', str),
        ('summary', str),
    ]
)

KNOWN_CATEGORIES = [
    KnownCategory(
        id='gates',
        title='Common Gates and Operations',
        summary='Unitary effects that can be put into quantum circuits.\n'
                '\n'
                'Note on terminology: a "gate" has no specified qubits, only '
                'an expected number of qubits, whereas an "operation" has '
                'specified qubits. "cirq.X" is a gate, "cirq.X(qubit)" is an '
                'operation.'),
    KnownCategory(
        id='devices',
        title='Devices and Qubits',
        summary='Classes for identifying the qubits and hardware you want to '
                'operate on.'),
    KnownCategory(
        id='advanced gates',
        title='Other Unitary Operations',
        summary='Some gates can be applied to arbitrary number of qubits.'),
    KnownCategory(
        id='noise',
        title='Noisy operations',
        summary='Non-unitary gates. Mixture gates are those that can be '
                'interpreted as applying a unitary for a fixed probability '
                'while channel encompasses the more general concept of a noisy '
                'open system quantum evolution.'),
    KnownCategory(
        id='developers',
        title='Advanced objects for developers',
        summary='For defining your own behavior.'),
    KnownCategory(
        id='decomposition',
        title='Decomposition',
        summary='Utilities for decomposing matrices and operations into '
                'gate sets.'),
    KnownCategory(
        id='linear algebra',
        title='Linear Algebra',
        summary='Utilities for working with vectors, matrices, and tensors.'),
    KnownCategory(
        id='circuits',
        title='Circuit Construction',
        summary='Objects for representing and building circuits.'),
    KnownCategory(
        id='simulation',
        title='Simulation',
        summary='Objects for simulating quantum circuits.'),
    KnownCategory(
        id='data collection',
        title='Sampling and Data Collection',
        summary='Objects for running quantum circuits against remote '
                'services.'),
    KnownCategory(
        id='protocols',
        title='Magic Method Protocols',
        summary='Utility methods for accessing and exposing generic '
                'functionality for gates, operations, and other types.'),
    KnownCategory(
        id='optimization',
        title='Optimization',
        summary='Objects for rewriting circuits.'),
    KnownCategory(
        id='basic value',
        title='Basic utilities and values',
        summary='Misc'),
    KnownCategory(
        id='experiments',
        title='Experiments',
        summary='Utilities for running experiments on simulators or hardware.'),
    KnownCategory(
        id='ion traps',
        title='Ion Traps',
        summary='Support for ion trap devices.'),
    KnownCategory(
        id='neutral atoms',
        title='Neutral Atoms',
        summary='Support for neutral atom devices.'),
    KnownCategory(
        id='google/service',
        title='Google Service',
        summary='Utilities for submitting work to remote machines.'),
    KnownCategory(
        id='testing',
        title='',
        summary='Utilities for writing unit tests involving cirq.'),
    KnownCategory(
        id='interop',
        title='Interop',
        summary='Code for interoperating with other frameworks.'),
    KnownCategory(
        id='schedules',
        title='Schedules',
        summary='Code for creating and working with schedules.'),
    KnownCategory(
        id='visualization',
        title='Plotting',
        summary='Utilities for plotting and visualizing data.'),
    KnownCategory(
        id='operator algebra',
        title='Operator Algebra',
        summary='Utilities for doing algebra with gates and operations.'),
    KnownCategory(
        id='deprecated',
        title='Deprecated',
        summary='Objects that exist but are being removed.'),
]


def main():
    categories = defaultdict(list)
    seen = set()
    for x in _find_public_cirq_members():
        if x.name in seen:
            continue
        seen.add(x.name)
        categories[x.doc_props.api_reference_category].append(x)

    expected_categories = {e.id for e in KNOWN_CATEGORIES}
    assert expected_categories == categories.keys(), (
        f'\n'
        f'MISSING DATA FOR DOCUMENTED CATEGORIES:\n'
        f'{sorted(categories.keys() - expected_categories)}\n'
        f'\n'
        f'HAVE DATA FOR EMPTY CATEGORIES:\n'
        f'{sorted(expected_categories - categories.keys())}\n'
    )


    print("""
.. currentmodule:: cirq

API Reference
=============

""")

    for known_category in KNOWN_CATEGORIES:
        print(known_category.title)
        print("'" * len(known_category.title))
        print()
        print(known_category.summary)
        print()
        print('.. autosummary::')
        print('    :toctree: generated/')
        print()
        for x in sorted(categories[known_category.id],
                        key=lambda e: (e.name.lower(), e.name)):
            print(f'    {x.scope}.{x.name}')
        print()
        print()


if __name__ == '__main__':
    main()
