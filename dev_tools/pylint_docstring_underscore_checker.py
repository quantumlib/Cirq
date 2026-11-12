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

"""Pylint checker and helpers for detecting unescaped underscore symbols in docstrings."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import TYPE_CHECKING

from pylint.checkers import BaseRawFileChecker

if TYPE_CHECKING:
    from astroid import nodes
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

_FENCED_CODE_BLOCK_RE = re.compile(r"(`{3,}|~{3,}).*?\1", flags=re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
# tensorflow_docs renders everything from `>>>` up to the next blank line as a code block,
# including `...` continuation lines and expected output lines.
_DOCTEST_BLOCK_RE = re.compile(r"^[ \t]*>>>.*(?:\n(?![ \t]*$).+)*", flags=re.MULTILINE)
_DOUBLE_BACKTICK_RE = re.compile(r"``(?:(?!``)[^\n])+``")
_SINGLE_BACKTICK_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+")
# Pandoc-style single-line inline math: opening $ followed by non-space/non-$,
# closing $ preceded by non-space/non-\ and not followed by a digit or $.
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?![\s$])[^$\n]*?(?<![\s\\])\$(?![0-9$])")

_SECTION_HEADER_RE = re.compile(
    r"^(Args|Arguments|Attributes|Parameters|Returns|Yields|Raises|Example|Examples|Usage):\s*$"
)
_PARAM_SECTION_NAMES = {"Args", "Arguments", "Attributes", "Parameters"}
# Note: Google-style `Example:`/`Examples:`/`Usage:` sections often contain un-fenced
# 4-space-indented code directly under the header. Masking all 4-space-indented lines in
# these sections avoids false positives on such code, at the cost of also masking prose in
# those sections (which `tensorflow_docs` renders as normal Markdown).
_CODE_SECTION_NAMES = {"Example", "Examples", "Usage"}
_PARAM_DEF_RE = re.compile(r"^\*{0,2}[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\([^)]*\))?\s*:")
_STRING_TOKEN_QUOTE_RE = re.compile(r"^([rRuUbBfF]*(\"\"\"|\'\'\'|\"|\'))(.*)\2$", flags=re.DOTALL)


def _splitlines_lf(text: str, *, keepends: bool = False) -> list[str]:
    r"""Like `str.splitlines`, but splits only on `\n` to match Python `ast`/`tokenize`."""
    if not text:
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
        return [p + "\n" for p in parts] if keepends else parts
    if keepends:
        return [p + "\n" for p in parts[:-1]] + [parts[-1]]
    return parts


def _mask_with_spaces(match: re.Match[str]) -> str:
    """Replaces matched characters with spaces while preserving newlines for line numbers."""
    return re.sub(r"[^\n]", " ", match.group(0))


def mask_docstring_code_and_math(docstring: str) -> str:
    """Masks out code blocks, doctests, inline code, URLs, LaTeX math, and `Args:` names.

    Newlines and character positions within unmasked text are preserved so that
    line numbers and column offsets remain accurate.
    """
    lines = _splitlines_lf(docstring, keepends=True)
    indents = [len(line) - len(line.lstrip(" ")) for line in lines[1:] if line.strip()]
    base_indent = min(indents) if indents else 0

    # 1. Mask multi-line / block constructs first.
    masked = _FENCED_CODE_BLOCK_RE.sub(_mask_with_spaces, docstring)
    masked = _DISPLAY_MATH_RE.sub(_mask_with_spaces, masked)
    masked = _DOCTEST_BLOCK_RE.sub(_mask_with_spaces, masked)

    # 2. Mask indented code blocks and parameter names inside Args/Attributes sections.
    out_lines: list[str] = []
    in_indented_block = False
    in_param_section = False
    in_code_section = False
    prev_ended_with_colon = False
    saw_blank_since_colon = False

    for idx, line in enumerate(_splitlines_lf(masked, keepends=True)):
        stripped = line.strip()
        if not stripped:
            if prev_ended_with_colon:
                saw_blank_since_colon = True
            out_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip(" "))
        rel_indent = indent - base_indent if idx > 0 else 0

        if rel_indent <= 0:
            in_indented_block = False
            in_code_section = False
            header_match = _SECTION_HEADER_RE.match(stripped)
            if header_match:
                section_name = header_match.group(1)
                in_param_section = section_name in _PARAM_SECTION_NAMES
                in_code_section = section_name in _CODE_SECTION_NAMES
                prev_ended_with_colon = False
                saw_blank_since_colon = False
                out_lines.append(line)
                continue
            if stripped.endswith(":"):
                in_param_section = False

        if in_indented_block:
            if rel_indent >= 4:
                out_lines.append(re.sub(r"[^\n]", " ", line))
                continue
            in_indented_block = False

        if (
            in_code_section or (prev_ended_with_colon and saw_blank_since_colon)
        ) and rel_indent >= 4:
            in_indented_block = True
            out_lines.append(re.sub(r"[^\n]", " ", line))
            continue

        if in_param_section and rel_indent >= 2 and _PARAM_DEF_RE.match(stripped):
            colon_idx = line.index(":")
            line = " " * colon_idx + line[colon_idx:]

        prev_ended_with_colon = stripped.endswith(":")
        saw_blank_since_colon = False
        out_lines.append(line)

    masked = "".join(out_lines)

    # 3. Mask single-line inline constructs.
    masked = _DOUBLE_BACKTICK_RE.sub(_mask_with_spaces, masked)
    masked = _SINGLE_BACKTICK_RE.sub(_mask_with_spaces, masked)
    masked = _URL_RE.sub(_mask_with_spaces, masked)
    masked = _INLINE_MATH_RE.sub(_mask_with_spaces, masked)
    return masked


def find_unescaped_underscore_symbols_in_docstring(
    docstring: str, line_map: list[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    """Finds unescaped underscore-delimited symbols in a docstring.

    Args:
        docstring: The docstring text to inspect.
        line_map: List of `(source_lineno, source_col_offset)` for each line in `docstring`.

    Returns:
        A list of `(lineno, col_offset, symbol)` tuples for each unescaped symbol found.
    """
    masked = mask_docstring_code_and_math(docstring)
    results: list[tuple[int, int, str]] = []
    for line_idx, line in enumerate(_splitlines_lf(masked)):
        lineno, col_base = line_map[line_idx]
        for match in UNESCAPED_UNDERSCORE_PATTERN.finditer(line):
            results.append((lineno, col_base + match.start(), match.group(1)))
    return results


def _extract_docstring_nodes(tree: ast.AST) -> list[ast.Constant]:
    """Extracts AST Constant string nodes for module/class/function/attribute docstrings and
    `document()` calls.
    """
    nodes: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                nodes.append(node.body[0].value)
            if isinstance(node, (ast.Module, ast.ClassDef)):
                for i in range(len(node.body) - 1):
                    stmt = node.body[i]
                    next_stmt = node.body[i + 1]
                    if (
                        isinstance(stmt, (ast.Assign, ast.AnnAssign))
                        and isinstance(next_stmt, ast.Expr)
                        and isinstance(next_stmt.value, ast.Constant)
                        and isinstance(next_stmt.value.value, str)
                    ):
                        nodes.append(next_stmt.value)
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "document":
                if (
                    len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    nodes.append(node.args[1])
                for kw in node.keywords:
                    if (
                        kw.arg == "doc_string"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        nodes.append(kw.value)
    return nodes


def _byte_col_to_char_col(line: str, byte_col: int) -> int:
    """Converts an AST UTF-8 byte column offset to a character offset matching `tokenize`."""
    if line.isascii():
        return byte_col
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def _get_raw_docstring_and_line_map(
    source_lines: list[str], string_tokens: list[tokenize.TokenInfo], node: ast.Constant
) -> tuple[str, list[tuple[int, int]]]:
    """Reconstructs raw text and `(source_lineno, source_col_offset)` per line for `node`."""
    start = (node.lineno, _byte_col_to_char_col(source_lines[node.lineno - 1], node.col_offset))
    end_lineno = node.end_lineno or node.lineno
    end = (
        end_lineno,
        _byte_col_to_char_col(source_lines[end_lineno - 1], node.end_col_offset or 0),
    )
    raw_parts: list[str] = []
    line_map: list[tuple[int, int]] = []

    for tok in string_tokens:
        if tok.start >= start and tok.end <= end:
            m = _STRING_TOKEN_QUOTE_RE.match(tok.string)
            if not m:  # pragma: no cover
                continue
            prefix_len = len(m.group(1))
            inner = m.group(3)
            inner_lines = _splitlines_lf(inner, keepends=True)
            if not inner_lines:
                inner_lines = [""]
            for idx, line_text in enumerate(inner_lines):
                src_line = tok.start[0] + idx
                src_col = (tok.start[1] + prefix_len) if idx == 0 else 0
                if idx == 0 and raw_parts and not raw_parts[-1].endswith("\n"):
                    raw_parts.append("\n")
                raw_parts.append(line_text)
                line_map.append((src_line, src_col))

    if not line_map:  # pragma: no cover
        assert isinstance(node.value, str)
        num_lines = max(1, len(_splitlines_lf(node.value)))
        return node.value, [(start[0] + i, start[1] if i == 0 else 0) for i in range(num_lines)]
    return "".join(raw_parts), line_map


def find_unescaped_underscore_symbols_in_source(source: str) -> list[tuple[int, int, str]]:
    """Parses Python source code and checks all docstrings and `document()` calls.

    Args:
        source: The Python source code string.

    Returns:
        A sorted list of `(lineno, col_offset, symbol)` tuples for each unescaped symbol found.
    """
    tree = ast.parse(source)
    source_lines = source.split("\n")
    string_tokens = [
        tok
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.STRING
    ]
    findings: list[tuple[int, int, str]] = []

    for doc_node in _extract_docstring_nodes(tree):
        raw_text, line_map = _get_raw_docstring_and_line_map(source_lines, string_tokens, doc_node)
        findings.extend(find_unescaped_underscore_symbols_in_docstring(raw_text, line_map))

    findings.sort()
    return findings


class DocstringUnderscoreChecker(BaseRawFileChecker):
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
    options = ()

    def process_module(self, node: nodes.Module) -> None:
        """Checks docstrings in the module for unescaped underscore symbols."""
        if not self.linter.is_message_enabled("unescaped-underscore-in-docstring"):
            return
        with node.stream() as stream:
            source = stream.read().decode("utf-8", errors="replace")
        try:
            findings = find_unescaped_underscore_symbols_in_source(source)
        except (SyntaxError, tokenize.TokenError):  # pragma: no cover
            return
        for lineno, col_offset, symbol in findings:
            self.add_message(
                "unescaped-underscore-in-docstring",
                line=lineno,
                col_offset=col_offset,
                args=(symbol,),
            )


def register(linter: PyLinter) -> None:
    """Register this checker to pylint."""
    linter.register_checker(DocstringUnderscoreChecker(linter))
