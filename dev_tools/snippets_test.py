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

"""Tests for executable snippets in documentation.

This tests runs code snippets that are executable in `.md` and `.rst`
documentation. It covers all such files under the docs directory, as well as
the top level README file.

In addition to checking that the code executes:

    * The test looks for comments of the form `# prints` and then the test
      checks that the result of the code snippets commented out code after
      that print statement.  So if the snippet is

          print('foo')
          # prints
          # foo

      Then this checks that the print statement indeed prints 'foo'.  Note that
      leading spaces are ignored.  If there are any characters after `# prints`,
      like for instance `# prints something like` then this comparison is
      not done. This is useful for documenting code that does print but
      the output is non-deterministic.

    * The test looks for substitutions that will be applied to the snippets
      before running the code. This is useful if a documentation example has
      a very long runtime, but can be made shorter by changing some variables
      (like number of qubits or number of repetitions).  For `.md` files the
      substitution is of the form

            <!---test_substitution
            pattern
            substitution
            --->

      and for `.rst` the substitution is of the form

            .. test-substitution::
                pattern
                substitution

      where pattern is the regex matching pattern (passed to re.compile) and
      substitution is the replacement string.
"""
import inspect
import sys
from typing import Any, Dict, List, Optional, Pattern, Tuple, Iterator

import os
import pathlib
import re

import pytest

import cirq


def test_can_run_readme_code_snippets():
    # Get the contents of the README.md file at the project root.
    readme_path = 'README.rst'
    assert readme_path is not None

    assert_file_has_working_code_snippets(readme_path, assume_import=False)


def find_docs_code_snippets_paths() -> Iterator[str]:
    docs_folder = pathlib.Path(__file__).parent
    for filename in docs_folder.rglob('*.md'):
        yield str(filename.relative_to(docs_folder))
    for filename in docs_folder.rglob('*.rst'):
        yield str(filename.relative_to(docs_folder))


@pytest.mark.parametrize('path', find_docs_code_snippets_paths())
def test_can_run_docs_code_snippets(path):
    docs_folder = os.path.dirname(__file__)
    assert_file_has_working_code_snippets(os.path.join(docs_folder, path), assume_import=True)


def find_code_snippets(pattern: str, content: str) -> List[Tuple[str, int]]:
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
    newlines = re.finditer("\n", content)
    snippets = []
    current_line = 1
    for match in matches:
        for newline in newlines:
            current_line += 1
            if newline.start() >= match.start():
                snippets.append((match.group(1), current_line))
                break
    return snippets


def find_markdown_code_snippets(content: str) -> List[Tuple[str, int]]:
    return find_code_snippets("\n```python(.*?)\n```\n", content)


def find_markdown_test_overrides(content: str) -> List[Tuple[Pattern, str]]:
    test_sub_text = find_code_snippets("<!---test_substitution\n(.*?)--->", content)
    substitutions = [line.split('\n')[:-1] for line, _ in test_sub_text]
    return [(re.compile(match), sub) for match, sub in substitutions]


def apply_overrides(content: str, overrides: List[Tuple[Pattern, str]]) -> str:
    override_content = content
    for pattern, sub in overrides:
        override_content = re.sub(pattern, sub, override_content)
    return override_content


def deindent_snippet(snippet: str) -> str:
    deindented_lines = []
    indentation_amount = None

    for line in snippet.split('\n'):
        # The first non-empty line determines the indentation level.
        if indentation_amount is None and re.match(r'\s*\S', line):
            leading_whitespace = re.match(r'\s*', line)
            if leading_whitespace:
                indentation_amount = len(leading_whitespace.group(0))

        if line:
            deindented_lines.append(line[indentation_amount:])
        else:
            deindented_lines.append(line)
    return '\n'.join(deindented_lines)


def find_rst_code_snippets(content: str) -> List[Tuple[str, int]]:
    snippets = find_code_snippets(
        r'\n.. code-block:: python\n(?:\s+:.*?\n)*\n(.*?)(?:\n\S|\Z)', content
    )
    return [(deindent_snippet(content), line_number) for content, line_number in snippets]


def find_rst_test_overrides(content: str) -> List[Tuple[Pattern, str]]:
    # Find ".. test-substitution::"
    test_sub_text = find_code_snippets(r'.. test-substitution::\n(([^\n]*\n){2})', content)
    substitutions = [line.split('\n')[:-1] for line, _ in test_sub_text]
    return [(re.compile(match.lstrip()), sub.lstrip()) for match, sub in substitutions]


def test_find_rst_code_snippets():
    snippets = find_rst_code_snippets(
        """
A 3 by 3 grid of qubits using

.. code-block:: python

    print("hello world")

The next level up.

.. code-block:: python
    :emphasize-lines: 3,5

    print("hello 1")

    for i in range(10):
        print(f"hello {i}")

More text.

.. code-block:: python

    print("last line")
"""
    )

    assert snippets == [
        ('print("hello world")\n', 4),
        ('print("hello 1")\n\nfor i in range(10):\n    print(f"hello {i}")\n', 10),
        ('print("last line")\n', 20),
    ]


def test_find_rst_overrides():
    overrides = find_rst_test_overrides(
        """
A 3 by 3 grid of qubits using

.. code-block:: python

    print("hello world")
    print("golden")

.. test-substitution::
    hello world
    goodbye cruel world

.. test-substitution::
    golden
    yellow
"""
    )
    assert len(overrides) == 2
    assert overrides[0][0].match('hello world')
    assert overrides[1][0].match('golden')
    assert overrides[0][1] == 'goodbye cruel world'
    assert overrides[1][1] == 'yellow'


def test_apply_rst_overrides():
    content = """
A 3 by 3 grid of qubits using

.. code-block:: python

    print("hello world")
    print("golden")

.. test-substitution::
    hello world
    goodbye cruel world

.. test-substitution::
    golden
    yellow
"""
    overrides = find_rst_test_overrides(content)
    print(overrides)
    assert (
        apply_overrides(content, overrides)
        == """
A 3 by 3 grid of qubits using

.. code-block:: python

    print("goodbye cruel world")
    print("yellow")

.. test-substitution::
    goodbye cruel world
    goodbye cruel world

.. test-substitution::
    yellow
    yellow
"""
    )


def test_find_markdown_code_snippets():
    snippets = find_markdown_code_snippets(
        """
A 3 by 3 grid of qubits using

```python
print("hello world")
```

The next level up.

```python
print("hello 1")

for i in range(10):
    print(f"hello {i}")
```

More text.

```python
print("last line")
```
"""
    )

    assert snippets == [
        ('\nprint("hello world")', 4),
        ('\nprint("hello 1")\n\nfor i in range(10):\n    print(f"hello {i}")', 10),
        ('\nprint("last line")', 19),
    ]


def test_find_markdown_test_overrides():
    overrides = find_markdown_test_overrides(
        """
A 3 by 3 grid of qubits using

```python
print("hello world")
```
<!---test_substitution
hello
goodbye
--->
<!---test_substitution
world
universe
--->
"""
    )

    assert len(overrides) == 2
    assert overrides[0][0].match('hello')
    assert overrides[1][0].match('world')
    assert overrides[0][1] == 'goodbye'
    assert overrides[1][1] == 'universe'


def test_apply_overrides_markdown():
    content = """
A 3 by 3 grid of qubits using

```python
print("hello world")
```
<!---test_substitution
hello
goodbye
--->
<!---test_substitution
world
universe
--->
"""
    overrides = find_markdown_test_overrides(content)
    assert (
        apply_overrides(content, overrides)
        == """
A 3 by 3 grid of qubits using

```python
print("goodbye universe")
```
<!---test_substitution
goodbye
goodbye
--->
<!---test_substitution
universe
universe
--->
"""
    )


def assert_file_has_working_code_snippets(path: str, assume_import: bool):
    """Checks that code snippets in a file actually run."""

    with open(path, encoding='utf-8') as f:
        content = f.read()

    # Find snippets of code, and execute them. They should finish.
    if path.endswith('.md'):
        overrides = find_markdown_test_overrides(content)
        content = apply_overrides(content, overrides)
        snippets = find_markdown_code_snippets(content)
    else:
        overrides = find_rst_test_overrides(content)
        content = apply_overrides(content, overrides)
        snippets = find_rst_code_snippets(content)
    assert_code_snippets_run_in_sequence(snippets, assume_import)


def assert_code_snippets_run_in_sequence(snippets: List[Tuple[str, int]], assume_import: bool):
    """Checks that a sequence of code snippets actually run.

    State is kept between snippets. Imports and variables defined in one
    snippet will be visible in later snippets.
    """

    state: Dict[str, Any] = {}

    if assume_import:
        exec('import cirq', state)

    for content, line_number in snippets:
        assert_code_snippet_executes_correctly(content, state, line_number)


def _canonicalize_printed_line_chunk(chunk: str) -> str:
    chunk = ' ' + chunk + ' '

    # Reduce trailing '.0' at end of number.
    chunk = chunk.replace('.0-', '. -')
    chunk = chunk.replace('.0+', '. +')

    # Remove leading spacing.
    while '[ ' in chunk:
        chunk = chunk.replace('[ ', '[')

    # Remove sign before zero.
    chunk = chunk.replace('-0 ', '+0 ')
    chunk = chunk.replace('-0. ', '+0. ')
    chunk = chunk.replace('-0j', '+0j')
    chunk = chunk.replace('-0.j', '+0.j')

    # Remove possibly-redundant + sign.
    chunk = chunk.replace(' +0. ', ' 0. ')
    chunk = chunk.replace(' +0.j', ' 0.j')

    # Remove double-spacing.
    while '  ' in chunk:
        chunk = chunk.replace('  ', ' ')

    # Remove spaces before imaginary unit.
    while ' j' in chunk:
        chunk = chunk.replace(' j', 'j')

    # Remove padding spaces.
    chunk = chunk.strip()

    if chunk.startswith('+'):
        chunk = chunk[1:]

    return chunk


def canonicalize_printed_line(line: str) -> str:
    """Remove minor variations between outputs on some systems.

    Basically, numpy is extremely inconsistent about where it puts spaces and
    minus signs on 0s. This method goes through the line looking for stuff
    that looks like it came from numpy, and if so then strips out spacing and
    turns signed zeroes into just zeroes.

    Args:
        line: The line to canonicalize.

    Returns:
        The canonicalized line.
    """
    prev_end = 0
    result = []
    for match in re.finditer(r"\[([^\]]+\.[^\]]*)\]", line):
        start = match.start() + 1
        end = match.end() - 1
        result.append(line[prev_end:start])
        result.append(_canonicalize_printed_line_chunk(line[start:end]))
        prev_end = end
    result.append(line[prev_end:])
    return ''.join(result).rstrip()


def test_canonicalize_printed_line():
    x = 'first [-0.5-0.j   0. -0.5j] then [-0.  0.]'
    assert canonicalize_printed_line(x) == ('first [-0.5+0.j 0. -0.5j] then [0. 0.]')

    a = '[-0.5-0.j   0. -0.5j  0. -0.5j -0.5+0.j ]'
    b = '[-0.5-0. j  0. -0.5j  0. -0.5j -0.5+0. j]'
    assert canonicalize_printed_line(a) == canonicalize_printed_line(b)

    assert len({canonicalize_printed_line(e) for e in ['[2.2]', '[+2.2]', '[ 2.2]']}) == 1

    assert len({canonicalize_printed_line(e) for e in ['[-0.]', '[+0.]', '[ 0.]', '[0.]']}) == 1

    a = '[[ 0.+0.j 1.+0.j]'
    b = '[[0.+0.j 1.+0.j]'
    assert canonicalize_printed_line(a) == canonicalize_printed_line(b)


def assert_code_snippet_executes_correctly(
    snippet: str, state: Dict, line_number: Optional[int] = None
):
    """Executes a snippet and compares output / errors to annotations."""

    raises_annotation = re.search(r"# raises\s*(\S*)", snippet)
    if raises_annotation is None:
        before = snippet
        after = None
        expected_failure = None
    else:
        before = snippet[: raises_annotation.start()]
        after = snippet[raises_annotation.start() :]
        expected_failure = raises_annotation.group(1)
        if not expected_failure:
            raise AssertionError('No error type specified for # raises line.')

    assert_code_snippet_runs_and_prints_expected(before, state, line_number)
    if expected_failure is not None:
        assert after is not None
        assert_code_snippet_fails(after, state, expected_failure)


def assert_code_snippet_runs_and_prints_expected(
    snippet: str, state: Dict, line_number: Optional[int] = None
):
    """Executes a snippet and compares captured output to annotated output."""
    output_lines: List[str] = []
    expected_outputs = find_expected_outputs(snippet)

    def print_capture(*values, sep=' '):
        output_lines.extend(sep.join(str(e) for e in values).split('\n'))

    state['print'] = print_capture
    try:
        exec(snippet, state)

        assert_expected_lines_present_in_order(expected_outputs, output_lines)
    except AssertionError as ex:
        # pylint: disable=consider-using-f-string
        new_msg = ex.args[0] + '\n\nIn snippet{}:\n{}'.format(
            "" if line_number is None else " (line {})".format(line_number), _indent([snippet])
        )
        ex.args = (new_msg,) + tuple(ex.args[1:])
        raise


def assert_code_snippet_fails(snippet: str, state: Dict, expected_failure_type: str):
    try:
        exec(snippet, state)
    except Exception as ex:
        actual_failure_types = [e.__name__ for e in inspect.getmro(type(ex))]
        if expected_failure_type not in actual_failure_types:
            # pylint: disable=consider-using-f-string
            raise AssertionError(
                'Expected snippet to raise a {}, but it raised a {}.'.format(
                    expected_failure_type, ' -> '.join(actual_failure_types)
                )
            )
        return

    raise AssertionError('Expected snippet to fail, but it ran to completion.')


def assert_expected_lines_present_in_order(expected_lines: List[str], actual_lines: List[str]):
    """Checks that all expected lines are present.

    It is permitted for there to be extra actual lines between expected lines.
    """
    expected_lines = [canonicalize_printed_line(e) for e in expected_lines]
    actual_lines = [canonicalize_printed_line(e) for e in actual_lines]

    i = 0
    for expected in expected_lines:
        while i < len(actual_lines) and actual_lines[i] != expected:
            i += 1

        assert i < len(actual_lines), (
            'Missing expected line: {!r}\n'
            '\n'
            'Actual lines:\n'
            '{}\n'
            '\n'
            'Expected lines:\n'
            '{}\n'
            '\n'
            'Highlighted Differences:\n'
            '{}\n'
            ''.format(
                expected,
                _indent(actual_lines),
                _indent(expected_lines),
                _indent(
                    [
                        cirq.testing.highlight_text_differences(
                            '\n'.join(actual_lines), '\n'.join(expected_lines)
                        )
                    ]
                ),
            )
        )
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
    continue_key = '# '
    expected = []

    printing = False
    for line in snippet.split('\n'):
        if printing:
            if line.startswith(continue_key) or line == continue_key.strip():
                rest = line[len(continue_key) :]
                expected.append(rest)
            else:
                printing = False
        # Matches '# print', '# prints', '# print:', and '# prints:'
        elif re.match(r'^#\s*prints?:?\s*$', line):
            printing = True

    return expected


def _indent(lines: List[str]) -> str:
    return '\t' + '\n'.join(lines).replace('\n', '\n\t')


def test_find_expected_outputs():
    assert (
        find_expected_outputs(
            """
# print
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
# prints
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
# print:
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
#print:
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
# prints:
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
# prints:
# abc

# def
    """
        )
        == ['abc']
    )

    assert (
        find_expected_outputs(
            """
lorem ipsum

# prints
#   abc

a wondrous collection

# prints
# def
# ghi
    """
        )
        == ['  abc', 'def', 'ghi']
    )

    assert (
        find_expected_outputs(
            """
a wandering adventurer

# prints something like
#  prints
#prints
# pants
# trance
    """
        )
        == []
    )


def test_assert_expected_lines_present_in_order():
    assert_expected_lines_present_in_order(expected_lines=[], actual_lines=[])

    assert_expected_lines_present_in_order(expected_lines=[], actual_lines=['abc'])

    assert_expected_lines_present_in_order(expected_lines=['abc'], actual_lines=['abc'])

    with pytest.raises(AssertionError):
        assert_expected_lines_present_in_order(expected_lines=['abc'], actual_lines=[])

    assert_expected_lines_present_in_order(
        expected_lines=['abc', 'def'], actual_lines=['abc', 'def']
    )

    assert_expected_lines_present_in_order(
        expected_lines=['abc', 'def'], actual_lines=['abc', 'interruption', 'def']
    )

    with pytest.raises(AssertionError):
        assert_expected_lines_present_in_order(
            expected_lines=['abc', 'def'], actual_lines=['def', 'abc']
        )

    assert_expected_lines_present_in_order(expected_lines=['abc    '], actual_lines=['abc'])

    assert_expected_lines_present_in_order(expected_lines=['abc'], actual_lines=['abc      '])


def test_assert_code_snippet_executes_correctly():
    assert_code_snippet_executes_correctly("a = 1", {})
    assert_code_snippet_executes_correctly("a = b", {'b': 1})

    s = {}
    assert_code_snippet_executes_correctly("a = 1", s)
    assert s['a'] == 1

    with pytest.raises(NameError):
        assert_code_snippet_executes_correctly("a = b", {})

    with pytest.raises(SyntaxError):
        assert_code_snippet_executes_correctly("a = ;", {})

    assert_code_snippet_executes_correctly(
        """
print("abc")
# prints
# abc
        """,
        {},
    )

    if sys.version_info[0] >= 3:  # Our print capture only works in python 3.
        with pytest.raises(AssertionError):
            assert_code_snippet_executes_correctly(
                """
print("abc")
# prints
# def
                """,
                {},
            )

    assert_code_snippet_executes_correctly(
        """
# raises ZeroDivisionError
a = 1 / 0
    """,
        {},
    )

    assert_code_snippet_executes_correctly(
        """
# raises ArithmeticError
a = 1 / 0
        """,
        {},
    )

    assert_code_snippet_executes_correctly(
        """
# prints 123
print("123")

# raises SyntaxError
print "abc")
        """,
        {},
    )

    with pytest.raises(AssertionError):
        assert_code_snippet_executes_correctly(
            """
# raises ValueError
a = 1 / 0
            """,
            {},
        )

    with pytest.raises(AssertionError):
        assert_code_snippet_executes_correctly(
            """
# raises
a = 1
            """,
            {},
        )
