import sys
from typing import List, Dict

import os
import re

import pytest


def test_can_run_readme_code_snippets():
    # Get the contents of the README.md file at the project root.
    readme_path = os.path.join(
        os.path.dirname(__file__),  # Start at this file's directory.
        '..', '..', 'cirq', 'docs',  # Hacky check that we're under cirq/docs/.
        '..', '..', 'README.md')     # Get the readme two levels up.

    assert_file_has_working_code_snippets(readme_path, assume_import=False)


def test_can_run_docs_code_snippets():
    docs_folder = os.path.dirname(__file__)
    for filename in os.listdir(docs_folder):
        if not filename.endswith('.md'):
            continue
        path = os.path.join(docs_folder, filename)
        try:
            assert_file_has_working_code_snippets(path, assume_import=True)
        except:
            print('DOCS FILE:\n\t{}'.format(filename))
            raise


def assert_file_has_working_code_snippets(path: str, assume_import: bool):
    """Checks that code snippets in a file actually run."""

    with open(path, 'r') as f:
        content = f.read()

    # Find snippets of code, and execute them. They should finish.
    snippets = re.findall("\n```python(.*?)\n```\n",
                          content,
                          re.MULTILINE | re.DOTALL)
    assert_code_snippets_run_in_sequence(snippets, assume_import)


def assert_code_snippets_run_in_sequence(snippets: List[str],
                                         assume_import: bool):
    """Checks that a sequence of code snippets actually run.

    State is kept between snippets. Imports and variables defined in one
    snippet will be visible in later snippets.
    """

    state = {}

    if assume_import:
        exec('import cirq', state)

    for snippet in snippets:
        assert_code_snippet_runs_and_prints_expected(snippet, state)


def assert_code_snippet_runs_and_prints_expected(snippet: str, state: Dict):
    """Executes a snippet and compares captured output to annotated output."""
    output_lines = []
    expected_outputs = find_expected_outputs(snippet)

    def print_capture(*values, sep=' '):
        output_lines.extend(sep.join(str(e) for e in values).split('\n'))

    state['print'] = print_capture
    try:
        exec(snippet, state)

        # Can't re-assign print in python 2.
        if sys.version_info[0] >= 3:
            assert_expected_lines_present_in_order(expected_outputs,
                                                   output_lines)
    except:
        print('SNIPPET: \n' + _indent([snippet]))
        raise


def assert_expected_lines_present_in_order(expected_lines: List[str],
                                           actual_lines: List[str]):
    """Checks that all expected lines are present.

    It is permitted for there to be extra actual lines between expected lines.
    """
    expected_lines = [e.rstrip() for e in expected_lines]
    actual_lines = [e.rstrip() for e in actual_lines]

    i = 0
    for expected in expected_lines:
        while i < len(actual_lines) and actual_lines[i] != expected:
            i += 1

        if i >= len(actual_lines):
            print('ACTUAL LINES: \n' + _indent(actual_lines))
            print('EXPECTED LINES: \n' + _indent(expected_lines))
            raise AssertionError(
                'Missing expected line: {}'.format(expected))
        i += 1


def find_expected_outputs(snippet: str) -> List[str]:
    """Finds expected output lines within a snippet.

    Expected output must be annotated with a leading '# prints'.
    Lines below '# prints' must start with '# ' or be just '#' and not indent
    any more than that in order to add an expected line. As soon as a line
    breaks this pattern, expected output recording cuts off.

    Adding words after '# prints' causes the expected output lines to be
    skipped instead of included. For example, for random output say
    '# prints something like' to avoid checking the following lines.
    """
    start_key = '# prints'
    continue_key = '# '
    expected = []

    printing = False
    for line in snippet.split('\n'):
        if printing:
            if line.startswith(continue_key) or line == continue_key.strip():
                rest = line[len(continue_key):]
                expected.append(rest)
            else:
                printing = False
        elif line.startswith(start_key):
            rest = line[len(start_key):]
            if not rest.strip():
                printing = True

    return expected


def _indent(lines: List[str]) -> str:
    return '\t' + '\n'.join(lines).replace('\n', '\n\t')


def test_find_expected_outputs():
    assert find_expected_outputs("""
# prints
# abc

# def
    """) == ['abc']

    assert find_expected_outputs("""
lorem ipsum

# prints
#   abc

a wondrous collection

# prints
# def
# ghi
    """) == ['  abc', 'def', 'ghi']

    assert find_expected_outputs("""
a wandering adventurer

# prints something like
#  prints
#prints
# pants
# trance
    """) == []


def test_assert_expected_lines_present_in_order():
    assert_expected_lines_present_in_order(
        expected_lines=[],
        actual_lines=[])

    assert_expected_lines_present_in_order(
        expected_lines=[],
        actual_lines=['abc'])

    assert_expected_lines_present_in_order(
        expected_lines=['abc'],
        actual_lines=['abc'])

    with pytest.raises(AssertionError):
        assert_expected_lines_present_in_order(
            expected_lines=['abc'],
            actual_lines=[])

    assert_expected_lines_present_in_order(
        expected_lines=['abc', 'def'],
        actual_lines=['abc', 'def'])

    assert_expected_lines_present_in_order(
        expected_lines=['abc', 'def'],
        actual_lines=['abc', 'interruption', 'def'])

    with pytest.raises(AssertionError):
        assert_expected_lines_present_in_order(
            expected_lines=['abc', 'def'],
            actual_lines=['def', 'abc'])

    assert_expected_lines_present_in_order(
        expected_lines=['abc    '],
        actual_lines=['abc'])

    assert_expected_lines_present_in_order(
        expected_lines=['abc'],
        actual_lines=['abc      '])
