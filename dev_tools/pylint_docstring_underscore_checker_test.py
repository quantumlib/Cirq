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
from astroid import nodes, parse
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from pylint.testutils import CheckerTestCase, MessageTest

from dev_tools.pylint_docstring_underscore_checker import (
    DocstringUnderscoreChecker,
    find_unescaped_underscore_symbols,
)

REPO_ROOT = pathlib.Path(__file__).parent.parent


def _msg(node: nodes.NodeNG, line: int, symbol: str) -> MessageTest:
    return MessageTest(
        msg_id="unescaped-underscore-in-docstring",
        node=node,
        line=line,
        col_offset=node.col_offset,
        end_line=node.end_lineno,
        end_col_offset=node.end_col_offset,
        args=(symbol,),
    )


class TestDocstringUnderscoreChecker(CheckerTestCase):
    r"""Test the unescaped-underscore-in-docstring checker for Pylint."""

    CHECKER_CLASS = DocstringUnderscoreChecker

    def test_unescaped_symbols_reported_with_line_numbers(self) -> None:
        module = parse('"""Uses __init__ and _unitary_ without backticks."""\n')
        assert module.doc_node is not None
        with self.assertAddsMessages(
            _msg(module.doc_node, 1, "__init__"), _msg(module.doc_node, 1, "_unitary_")
        ):
            self.walk(module)

    def test_dotted_underscore_wrapped_symbols_are_flagged(self) -> None:
        module = parse('"""See __init__.py and _private_.attr for details."""\n')
        assert module.doc_node is not None
        with self.assertAddsMessages(
            _msg(module.doc_node, 1, "__init__"), _msg(module.doc_node, 1, "_private_")
        ):
            self.walk(module)

    def test_multiline_and_nested_definitions(self) -> None:
        source = (
            'class Outer:\n'
            '    """Summary line.\n'
            '\n'
            '    Details calling _private_helper on line 4.\n'
            '    """\n'
            '    async def inner(self) -> None:\n'
            '        """Inner method referencing __call__."""\n'
            '        "Bare non-docstring string with __ignored__ in body."\n'
        )
        module = parse(source)
        outer_cls = module.body[0]
        inner_fn = outer_cls.body[0]
        with self.assertAddsMessages(
            _msg(outer_cls.doc_node, 4, "_private_helper"), _msg(inner_fn.doc_node, 7, "__call__")
        ):
            self.walk(module)

    def test_document_call_and_attribute_docstrings(self) -> None:
        source = (
            'X = 1\n'
            '"""Attribute docstring with _attr_symbol_."""\n'
            'document(X, "Defaults to __debug__.")\n'
            '_doc.document(X, doc_string="Keyword _unitary_.")\n'
            'other_call(X, "Ignored __init__.")\n'
        )
        module = parse(source)
        attr_doc = module.body[1].value
        call1_arg = module.body[2].value.args[1]
        call2_kw = module.body[3].value.keywords[0].value
        with self.assertAddsMessages(
            _msg(attr_doc, 2, "_attr_symbol_"),
            _msg(call1_arg, 3, "__debug__"),
            _msg(call2_kw, 4, "_unitary_"),
        ):
            self.walk(module)

    def test_stray_currency_dollar_does_not_mask_symbols(self) -> None:
        module = parse('"""Price is $5 later _a_b_ and\nmore $6 after __init__."""\n')
        assert module.doc_node is not None
        with self.assertAddsMessages(
            _msg(module.doc_node, 1, "_a_b_"), _msg(module.doc_node, 2, "__init__")
        ):
            self.walk(module)

    def test_colon_in_prose_outside_args_is_flagged(self) -> None:
        module = parse('"""Note:\n    __init__: initializes the instance.\n"""\n')
        assert module.doc_node is not None
        with self.assertAddsMessages(_msg(module.doc_node, 2, "__init__")):
            self.walk(module)

    def test_leading_ellipsis_in_prose_is_flagged(self) -> None:
        module = parse('"""Summary.\n\n..._foo_bar_ at the start of a prose line.\n"""\n')
        assert module.doc_node is not None
        with self.assertAddsMessages(_msg(module.doc_node, 3, "_foo_bar_")):
            self.walk(module)

    def test_doctest_masks_output_until_blank_line_only(self) -> None:
        source = (
            '"""Summary.\n\n'
            '>>> get_name()\n'
            '_helper_fn_\n'
            '\n'
            'Prose after doctest referencing __init__.\n'
            '"""\n'
        )
        module = parse(source)
        assert module.doc_node is not None
        with self.assertAddsMessages(_msg(module.doc_node, 6, "__init__")):
            self.walk(module)

    def test_prose_after_indented_code_block_is_flagged(self) -> None:
        source = (
            '"""Summary.\n\n'
            '    Usage:\n\n'
            '        x = _in_code_block_\n'
            '    Prose after code block referencing __init__.\n'
            '"""\n'
        )
        module = parse(source)
        assert module.doc_node is not None
        with self.assertAddsMessages(_msg(module.doc_node, 6, "__init__")):
            self.walk(module)

    def test_multiline_joined_string_reports_node_lineno(self) -> None:
        source = (
            'def f():\n'
            '    (\n'
            '        "First line "\n'
            '        "second line referencing __init__."\n'
            '    )\n'
        )
        module = parse(source)
        fn = module.body[0]
        with self.assertAddsMessages(_msg(fn.doc_node, fn.doc_node.lineno, "__init__")):
            self.walk(module)

    def test_section_header_on_first_line_dedents_cleanly(self) -> None:
        doc = "Args:\n    _x: Parameter.\n    Calls __init__."
        assert find_unescaped_underscore_symbols(doc) == [(2, "__init__")]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param('def f():\n    pass\n', id="no-docstring"),
            pytest.param('def f():\n    """"""\n', id="empty-docstring"),
            pytest.param(
                '"""Uses `__init__`, `_unitary_`, and ``_private_helper`` in backticks."""\n',
                id="inline-backticks",
            ),
            pytest.param(
                '"""Calls `cirq._some_long_helper(\nx)` across lines."""\n',
                id="multiline-inline-backticks",
            ),
            pytest.param(
                '"""Code block:\n\n'
                '```python\ndef __init__(self):\n    self._unitary_()\n```\n"""\n',
                id="fenced-backtick-code-block",
            ),
            pytest.param(
                '"""Tilde code block:\n\n'
                '~~~python\ndef __init__(self):\n    self._unitary_()\n~~~\n"""\n',
                id="fenced-tilde-code-block",
            ),
            pytest.param(
                '"""Doctest:\n>>> obj._unitary_(\n...     __init__\n... )\n_helper_fn_\n"""\n',
                id="doctest-with-continuation-and-output",
            ),
            pytest.param(
                '"""Indented code block:\n\n'
                '    Usage:\n'
                '        def _default_do_a_using_b(self):\n'
                '            pass\n'
                '        x = _more_code_\n"""\n',
                id="indented-code-block",
            ),
            pytest.param(
                '"""Math expression $a_{i_1} + b_{j_2}$ and display $$x_{a_b}$$."""\n',
                id="latex-math",
            ),
            pytest.param(
                '"""URL (https://example.com/_foo_bar/__init__.py) and _module_name.attr and '
                '1_000_."""\n',
                id="urls-dotted-prefixes-and-numeric-literals",
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
            pytest.param('"""_a' + '_' * 40 + '-x."""\n', id="long-underscore-run-no-redos"),
        ],
    )
    def test_valid_contexts_not_flagged(self, source: str) -> None:
        with self.assertNoMessages():
            self.walk(parse(source))


def _lint(tmp_path: pathlib.Path, source: str) -> list[str]:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    rcfile = tmp_path / "pylintrc"
    rcfile.write_text("", encoding="utf-8")
    reporter = CollectingReporter()
    Run(
        [
            str(path),
            f"--rcfile={rcfile}",
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
