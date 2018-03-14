from typing import List, Dict

import os
import re


def test_readme_code_snippets_execute():
    # Get the contents of the README.md file at the project root.
    readme_path = os.path.join(
        os.path.dirname(__file__),  # Start at this file's directory.
        '..', '..', 'cirq', 'docs',  # Hacky check that we're under cirq/docs/.
        '..', '..', 'README.md')     # Get the readme two levels up.

    assert_file_has_working_code_snippets(readme_path, assume_import=False)


def test_docs_code_snippets_execute():
    docs_folder = os.path.dirname(__file__)
    for filename in os.listdir(docs_folder):
        if not filename.endswith('.md'):
            continue
        path = os.path.join(docs_folder, filename)
        try:
            assert_file_has_working_code_snippets(path, assume_import=True)
        except:
            print('Failing file: {}'.format(filename))
            raise


def assert_file_has_working_code_snippets(path: str, assume_import: bool):
    """Checks that code snippets in a file actually run."""

    try:
        with open(path, 'r') as f:
            content = f.read()
    except IOError:
        # File not found. Not great.. but no need to test that it runs!
        return

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
        expected_output_lines = []

        for line in snippet.split('\n'):
            if line.startswith('# prints:'):
                expected_output_lines.append(line[len('# prints:'):])

        assert_code_snippet_runs_and_prints_expected(snippet, state)
        try:
            exec(snippet, state)
        except:
            print('Failing snippet:\n{}'.format(snippet))
            raise


def assert_code_snippet_runs_and_prints_expected(snippet: str, state: Dict):
    """Executes a snippet and compares captured output to annotated output."""
    output_lines = []
    expected_outputs = find_expected_outputs(snippet)

    def print_capture(*values, sep=' '):
        output_lines.extend(sep.join(str(e) for e in values).split('\n'))

    state['print'] = print_capture
    try:
        exec(snippet, state)
        assert_expected_lines_present_in_order(expected_outputs, output_lines)
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

    Expected output must be annotated with a leading '# prints' or '# prints:'.
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
            if rest.startswith(':'):
                expected.append(rest[1:])
                printing = True
            elif not rest.strip():
                printing = True

    return expected


def _indent(lines: List[str]) -> str:
    return '\t' + '\n'.join(lines).replace('\n', '\n\t')
