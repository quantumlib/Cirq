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

from __future__ import annotations

from dev_tools import incremental_coverage


def test_determine_ignored_lines() -> None:
    f = incremental_coverage.determine_ignored_lines

    assert f("a = 0  # pragma: no cover") == {1}

    assert f("""
        a = 0  # pragma: no cover
        b = 0
    """) == {2}

    assert f("""
        a = 0
        b = 0  # pragma: no cover
    """) == {3}

    assert f("""
        a = 0  # pragma: no cover
        b = 0  # pragma: no cover
    """) == {2, 3}

    assert f("""
        if True:
            a = 0  # pragma: no cover

            b = 0
    """) == {3}

    assert f("""
        if True:
            # pragma: no cover
            a = 0

            b = 0
    """) == {3, 4, 5, 6, 7}

    assert f("""
        if True:
            # pragma: no cover
            a = 0

            b = 0
        stop = 1
    """) == {3, 4, 5, 6}

    assert f("""
        if True:
            # pragma: no cover
            a = 0

            b = 0
        else:
            c = 0
    """) == {3, 4, 5, 6}

    assert f("""
        if True:
            while False:
                # pragma: no cover
                a = 0

            b = 0
        else:
            c = 0  # pragma: no cover
    """) == {4, 5, 6, 9}

    assert f("""
        a = 2#pragma:no cover
        a = 3 #pragma:no cover
        a = 4# pragma:no cover
        a = 5#pragma :no cover
        a = 6#pragma: no cover
        a = 7#pragma: no cover\t
        a = 8#pragma:\tno cover\t

        b = 1 # no cover
        b = 2 # coverage: definitely
        b = 3 # lint: ignore
    """) == {2, 3, 4, 6, 7, 8}

    assert f("""
        if TYPE_CHECKING:
            import cirq
            import foo
        def bar(a: 'cirq.Circuit'):
            pass
    """) == {2, 3, 4}


def test_line_counts_as_uncovered_with_hashes() -> None:
    f = incremental_coverage.line_counts_as_uncovered
    # Simple line with hash in string.
    assert f("x = '#'", False) is True
    # Line with hash in string and a comment.
    assert f("x = '#' # comment", False) is True
    # Line that should be ignored (e.g., import).
    assert f("import os # comment", False) is False
    # Line with multiple hashes in string.
    assert f("x = '###'", False) is True
    # Check that it still ignores actual comments.
    assert f("x = 1 # some comment", False) is True

    # Coverage for is_from_cover_annotation_file=True
    assert f("! x = 1", True) is True
    assert f("  x = 1", True) is False
    assert f("! import os", True) is False

    # Coverage for tokenize.TokenError
    # An unclosed multi-line string will cause tokenize to raise TokenError
    assert f('""" # comment', False) is True
    assert f("''' # comment", False) is True

    # Additional coverage for line_content_counts_as_uncovered_manual
    assert f("def foo():", False) is False
    assert f("class Foo:", False) is False
    assert f("    ", False) is False


def test_is_applicable_python_file() -> None:
    f = incremental_coverage.is_applicable_python_file
    assert f("cirq/ops/gate.py") is True
    assert f("dev_tools/incremental_coverage.py") is False
    assert f("cirq/ops/gate_test.py") is True
    assert f("test.txt") is False
    assert f("benchmarks/perf.py") is False
