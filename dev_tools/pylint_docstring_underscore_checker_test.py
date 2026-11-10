# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

import pathlib
import tomllib

import pytest
from astroid import parse
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from pylint.testutils import CheckerTestCase, MessageTest

from dev_tools.pylint_docstring_underscore_checker import (
    DocstringUnderscoreChecker,
    find_unescaped_underscore_symbols_in_docstring,
    find_unescaped_underscore_symbols_in_file,
    find_unescaped_underscore_symbols_in_source,
)

REPO_ROOT = pathlib.Path(__file__).parent.parent


class TestDocstringUnderscoreChecker(CheckerTestCase):
    r"""Test the unescaped-underscore-in-docstring checker for Pylint."""

    CHECKER_CLASS = DocstringUnderscoreChecker

    def test_unescaped_symbols_reported_with_exact_columns(self) -> None:
        node = parse('"""Uses __init__ and _unitary_ without backticks."""\n')
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring", line=1, col_offset=8, args=("__init__",)
            ),
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=1,
                col_offset=21,
                args=("_unitary_",),
            ),
        ):
            self.checker.process_module(node)

    def test_multiline_and_nested_definitions(self) -> None:
        source = (
            'class Outer:\n'
            '    """Summary line.\n'
            '\n'
            '    Details calling _private_helper on line 4.\n'
            '    """\n'
            '    def inner(self) -> None:\n'
            '        """Inner method referencing __call__."""\n'
            '        "Bare non-docstring string with __ignored__ in body."\n'
        )
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=4,
                col_offset=20,
                args=("_private_helper",),
            ),
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=7,
                col_offset=36,
                args=("__call__",),
            ),
        ):
            self.checker.process_module(node)

    def test_document_call_and_attribute_docstrings(self) -> None:
        source = (
            'X = 1\n'
            '"""Attribute docstring with _attr_symbol_."""\n'
            'document(X, "Defaults to __debug__.")\n'
            '_doc.document(X, doc_string="Keyword _unitary_.")\n'
        )
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=2,
                col_offset=28,
                args=("_attr_symbol_",),
            ),
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=3,
                col_offset=25,
                args=("__debug__",),
            ),
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=4,
                col_offset=37,
                args=("_unitary_",),
            ),
        ):
            self.checker.process_module(node)

    def test_stray_currency_dollar_does_not_mask_symbols(self) -> None:
        source = '"""Price is $5 later _a_b_ and\nmore $6 after __init__."""\n'
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring", line=1, col_offset=21, args=("_a_b_",)
            ),
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=2,
                col_offset=14,
                args=("__init__",),
            ),
        ):
            self.checker.process_module(node)

    def test_colon_in_prose_outside_args_is_flagged(self) -> None:
        source = '"""Note:\n    __init__: initializes the instance.\n"""\n'
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring", line=2, col_offset=4, args=("__init__",)
            )
        ):
            self.checker.process_module(node)

    def test_leading_ellipsis_in_prose_is_flagged(self) -> None:
        source = '"""Summary.\n\n..._foo_bar_ at the start of a prose line.\n"""\n'
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=3,
                col_offset=3,
                args=("_foo_bar_",),
            )
        ):
            self.checker.process_module(node)

    def test_prose_after_indented_code_block_is_flagged(self) -> None:
        source = (
            '"""Summary:\n\n'
            '        x = _in_code_block_\n'
            '      Prose after code block referencing __init__.\n'
            '    Base indent line.\n'
            '"""\n'
        )
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=4,
                col_offset=41,
                args=("__init__",),
            )
        ):
            self.checker.process_module(node)

    def test_implicitly_concatenated_docstring_tokens(self) -> None:
        source = (
            'def f():\n'
            '    (\n'
            '        "First literal part "\n'
            '        "second part referencing __init__."\n'
            '    )\n'
        )
        node = parse(source)
        with self.assertAddsMessages(
            MessageTest(
                msg_id="unescaped-underscore-in-docstring",
                line=4,
                col_offset=33,
                args=("__init__",),
            )
        ):
            self.checker.process_module(node)

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param('def f():\n    """"""\n', id="empty-docstring"),
            pytest.param(
                '"""Uses `__init__`, `_unitary_`, and ``_private_helper`` in backticks."""\n',
                id="inline-backticks",
            ),
            pytest.param(
                '"""Code block:\n\n'
                '```python\ndef __init__(self):\n    self._unitary_()\n```\n"""\n',
                id="fenced-code-block",
            ),
            pytest.param(
                '"""Doctest:\n>>> obj._unitary_(\n...     __init__\n... )\n"""\n',
                id="doctest-with-continuation",
            ),
            pytest.param(
                '"""Indented code block:\n\n'
                '    Usage:\n'
                '        def _default_do_a_using_b(self):\n'
                '            pass\n"""\n',
                id="indented-code-block",
            ),
            pytest.param(
                '"""Program output:\n\n'
                '=== Example output ===\n'
                'Bases match: X_XXXXX_\n'
                'a: _1___1_1_1111_\n"""\n',
                id="example-output-block",
            ),
            pytest.param(
                '"""Math expression $a_{i_1} + b_{j_2}$ and display $$x_{a_b}$$."""\n',
                id="latex-math",
            ),
            pytest.param(
                '"""URL https://example.com/_foo_bar/__init__ and domain b_c_.com and 1_000_."""\n',
                id="urls-domains-and-numeric-literals",
            ),
            pytest.param(
                'r"""Backslash-escaped \\_unitary\\_ and \\__init\\__ in raw docstring."""\n',
                id="backslash-escaped",
            ),
            pytest.param(
                '"""Summary.\n\n'
                'Args:\n'
                '    _flatten_contents: If True, flattens.\n'
                '    _typed_arg (bool): Typed parameter.\n'
                '    **_extra_kwargs: Extra args.\n"""\n',
                id="args-section-parameter-names",
            ),
        ],
    )
    def test_valid_contexts_not_flagged(self, source: str) -> None:
        node = parse(source)
        with self.assertNoMessages():
            self.checker.process_module(node)

    def test_disabled_message_early_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = parse('"""Docstring with __init__."""\n')
        monkeypatch.setattr(self.linter, "is_message_enabled", lambda *_args, **_kwargs: False)
        with self.assertNoMessages():
            self.checker.process_module(node)


def _lint(tmp_path: pathlib.Path, source: str) -> list[str]:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    reporter = CollectingReporter()
    Run(
        [
            str(path),
            "--persistent=n",
            "--disable=all",
            "--enable=unescaped-underscore-in-docstring,useless-suppression",
            "--load-plugins=dev_tools.pylint_docstring_underscore_checker",
        ],
        reporter=reporter,
        exit=False,
    )
    return [m.symbol for m in reporter.messages]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            '# pylint: disable=unescaped-underscore-in-docstring\n"""Uses __init__."""\n', id="file"
        ),
        pytest.param(
            'def f():\n'
            '    # pylint: disable=unescaped-underscore-in-docstring\n'
            '    """Uses __init__."""\n',
            id="scope",
        ),
        pytest.param(
            'class K:\n'
            '    # pylint: disable-next=unescaped-underscore-in-docstring\n'
            '    """Uses __init__."""\n',
            id="next",
        ),
    ],
)
def test_pragma_suppresses_cleanly(tmp_path: pathlib.Path, source: str) -> None:
    assert _lint(tmp_path, source) == []


def test_unsuppressed_docstring_is_reported(tmp_path: pathlib.Path) -> None:
    assert _lint(tmp_path, '"""Uses __init__."""\n') == ["unescaped-underscore-in-docstring"]


def test_pyproject_enables_unescaped_underscore_checker() -> None:
    """Verify the plugin is loaded and enabled in pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pylint_cfg = pyproject["tool"]["pylint"]
    assert "dev_tools.pylint_docstring_underscore_checker" in pylint_cfg["main"]["load-plugins"]
    assert "unescaped-underscore-in-docstring" in pylint_cfg["messages_control"]["enable"]


def test_find_unescaped_underscore_symbols_in_source_offsets() -> None:
    assert find_unescaped_underscore_symbols_in_source(
        '"""Trailing symbol_name_ in prose."""\n'
    ) == [(1, 12, "symbol_name_")]


def test_find_unescaped_underscore_symbols_in_docstring_without_line_map() -> None:
    assert find_unescaped_underscore_symbols_in_docstring(
        "Summary.\nCalls _unitary_ directly."
    ) == [(2, 6, "_unitary_")]


def test_find_unescaped_underscore_symbols_in_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text('"""Module docstring referencing _private_helper."""\n', encoding="utf-8")
    assert find_unescaped_underscore_symbols_in_file(path) == [(1, 32, "_private_helper")]
