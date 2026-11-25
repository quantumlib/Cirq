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

"""Pylint checker for detecting unescaped underscore symbols in docstrings."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.checkers.utils import only_required_for_messages

if TYPE_CHECKING:
    from pylint.lint import PyLinter

# Matches unescaped symbols in docstring text that Markdown/DevSite renders as bold/italics:
# 1. Double-underscore (__foo__) or single-underscore-wrapped (_foo_) identifiers
# 2. Leading-underscore identifiers containing at least one more underscore (_foo_bar)
# 3. Trailing-underscore identifiers containing at least one internal underscore (foo_bar_)
# A trailing `.attr` / `.ext` is skipped only when the matched symbol does not end in `_`
# (CommonMark still renders `__init__.py` and `_private_.attr` with bold/italics).
UNESCAPED_UNDERSCORE_PATTERN = re.compile(
    r"(?<![`\w/\\-])"
    r"(_+[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*|[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+_+)"
    r"(?![`\w/\\-]|(?<!_)\.[a-zA-Z])"
)

# Masks fenced code blocks (``` / ~~~), display math ($$...$$), doctest blocks (including
# `...` continuations and output lines up to the next blank line), inline backticks, URLs,
# and single-line inline LaTeX math ($...$) while preserving newlines for line numbers.
_MASK_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})(?![`~])[^\n]*\n(?s:.*?)^[ \t]*(?P=fence)(?![`~])[ \t]*$"
    r"|\$\$(?s:.*?)\$\$"
    r"|^[ \t]*>>>[^\n]*(?:\n(?![ \t]*$)[^\n]+)*"
    r"|(?P<tick>`+)(?!`)(?:(?!\n[ \t]*\n)[^`])+?(?P=tick)(?!`)"
    r"|https?://[^\s)>\]\"'`]+"
    r"|(?<![\\$])\$(?![\s$])[^$\n]*?(?<![\s\\])\$(?![0-9$])",
    flags=re.MULTILINE,
)

_PARAM_SECTIONS = {"Args", "Arguments", "Attributes", "Parameters"}
_PROSE_SECTIONS = _PARAM_SECTIONS | {
    "Returns",
    "Yields",
    "Raises",
    "Note",
    "Notes",
    "See Also",
    "References",
}
# Note: Google-style `Example:`/`Examples:`/`Usage:` sections often contain un-fenced
# 4-space-indented code directly under the header. Masking 4-space-indented blocks in these
# sections avoids false positives on such code, at the cost of also masking prose there.
_CODE_SECTIONS = {"Example", "Examples", "Usage"}
_SECTION_RE = re.compile(rf"^({'|'.join(sorted(_PROSE_SECTIONS | _CODE_SECTIONS))}):\s*$")
_PARAM_NAME_RE = re.compile(r"^(\s*\*{0,2}[a-zA-Z_][a-zA-Z0-9_]*)(?=\s*(?:\(.*\))?\s*:)")


def _dedent_keep_lines(docstring: str) -> str:
    """Dedents a docstring like `inspect.cleandoc` while preserving exact line count."""
    lines = docstring.expandtabs().split("\n")
    indents = [len(line) - len(line.lstrip(" ")) for line in lines[1:] if line.strip()]
    if len(lines) > 1 and not lines[-1].strip() and lines[-1]:
        margin = min([len(lines[-1]), *indents])
    elif lines[0].strip() and _SECTION_RE.match(lines[0].strip()):
        margin = 0
    else:
        margin = min(indents, default=0)
    return "\n".join([lines[0].lstrip(" "), *(line[margin:] for line in lines[1:])])


def find_unescaped_underscore_symbols(docstring: str) -> list[tuple[int, str]]:
    """Finds unescaped underscore-delimited symbols in a docstring.

    Args:
        docstring: The docstring text to inspect.

    Returns:
        A list of `(line_offset, symbol)` tuples where `line_offset` is 0-indexed
        relative to the start of `docstring`.
    """
    masked = _MASK_RE.sub(
        lambda m: re.sub(r"[^\n]", " ", m.group(0)), _dedent_keep_lines(docstring)
    )
    section: str | None = None
    in_code = False
    after_blank = False
    findings: list[tuple[int, str]] = []

    for idx, line in enumerate(masked.split("\n")):
        stripped = line.strip()
        if not stripped:
            after_blank = True
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            m_sec = _SECTION_RE.match(stripped)
            section = m_sec.group(1) if m_sec else None
            in_code = after_blank = False
            if m_sec:
                continue

        code_indent = (
            12
            if section in (_PARAM_SECTIONS | {"Raises"})
            else 8 if section in _PROSE_SECTIONS else 4
        )
        if not (in_code and indent >= code_indent):
            in_code = indent >= code_indent and (after_blank or section in _CODE_SECTIONS)
        after_blank = False
        if in_code:
            continue

        if section in _PARAM_SECTIONS and 2 <= indent <= 4:
            line = _PARAM_NAME_RE.sub(lambda m: " " * len(m.group(1)), line)

        findings.extend(
            (idx, match.group(1)) for match in UNESCAPED_UNDERSCORE_PATTERN.finditer(line)
        )
    return findings


class DocstringUnderscoreChecker(BaseChecker):
    """Pylint checker for unescaped underscore-delimited symbols in docstrings.

    This check can be disabled with `# pylint: disable=unescaped-underscore-in-docstring`.
    """

    name = "unescaped-underscore-in-docstring"
    msgs = {
        "C0002": (
            (
                "Unescaped symbol '%s' in docstring; wrap code symbols in backticks or use "
                "*emphasis*"
            ),
            "unescaped-underscore-in-docstring",
            (
                "Wrap underscore-delimited symbols (such as `__init__`, `_unitary_`, or "
                "`_private_helper`) in backticks in docstrings so Markdown does not render "
                "them as italics or bold, or use `*...*` / `**...**` when emphasis is intended."
            ),
        )
    }

    def _check_const(self, doc_node: nodes.Const | None) -> None:
        if not doc_node or not isinstance(doc_node.value, str):
            return
        span = (doc_node.end_lineno or doc_node.lineno) - doc_node.lineno + 1
        same_lines = len(doc_node.value.split("\n")) == span
        for line_offset, symbol in find_unescaped_underscore_symbols(doc_node.value):
            lineno = (doc_node.lineno + line_offset) if same_lines else doc_node.lineno
            self.add_message(
                "unescaped-underscore-in-docstring", node=doc_node, line=lineno, args=(symbol,)
            )

    def _check_scope(self, node: nodes.Module | nodes.ClassDef) -> None:
        self._check_const(node.doc_node)
        for s1, s2 in zip(node.body, node.body[1:]):
            if (
                isinstance(s1, (nodes.Assign, nodes.AnnAssign))
                and isinstance(s2, nodes.Expr)
                and isinstance(s2.value, nodes.Const)
            ):
                self._check_const(s2.value)

    @only_required_for_messages("unescaped-underscore-in-docstring")
    def visit_module(self, node: nodes.Module) -> None:
        """Checks module and module-attribute docstrings."""
        self._check_scope(node)

    @only_required_for_messages("unescaped-underscore-in-docstring")
    def visit_classdef(self, node: nodes.ClassDef) -> None:
        """Checks class and class-attribute docstrings."""
        self._check_scope(node)

    @only_required_for_messages("unescaped-underscore-in-docstring")
    def visit_functiondef(self, node: nodes.FunctionDef | nodes.AsyncFunctionDef) -> None:
        """Checks function or method docstrings."""
        self._check_const(node.doc_node)

    visit_asyncfunctiondef = visit_functiondef

    @only_required_for_messages("unescaped-underscore-in-docstring")
    def visit_call(self, node: nodes.Call) -> None:
        """Checks `document(...)` documentation strings."""
        func_name = (
            node.func.name
            if isinstance(node.func, nodes.Name)
            else node.func.attrname if isinstance(node.func, nodes.Attribute) else None
        )
        if func_name == "document":
            for arg in [
                *node.args[1:2],
                *(kw.value for kw in node.keywords if kw.arg == "doc_string"),
            ]:
                if isinstance(arg, nodes.Const):
                    self._check_const(arg)


def register(linter: PyLinter) -> None:
    """Register this checker to pylint."""
    linter.register_checker(DocstringUnderscoreChecker(linter))
