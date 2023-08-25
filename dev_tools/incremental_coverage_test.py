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

from dev_tools import incremental_coverage


def test_determine_ignored_lines():
    f = incremental_coverage.determine_ignored_lines

    assert f("a = 0  # pragma: no cover") == {1}

    assert (
        f(
            """
        a = 0  # pragma: no cover
        b = 0
    """
        )
        == {2}
    )

    assert (
        f(
            """
        a = 0
        b = 0  # pragma: no cover
    """
        )
        == {3}
    )

    assert (
        f(
            """
        a = 0  # pragma: no cover
        b = 0  # pragma: no cover
    """
        )
        == {2, 3}
    )

    assert (
        f(
            """
        if True:
            a = 0  # pragma: no cover

            b = 0
    """
        )
        == {3}
    )

    assert (
        f(
            """
        if True:
            # pragma: no cover
            a = 0

            b = 0
    """
        )
        == {3, 4, 5, 6, 7}
    )

    assert (
        f(
            """
        if True:
            # pragma: no cover
            a = 0

            b = 0
        stop = 1
    """
        )
        == {3, 4, 5, 6}
    )

    assert (
        f(
            """
        if True:
            # pragma: no cover
            a = 0

            b = 0
        else:
            c = 0
    """
        )
        == {3, 4, 5, 6}
    )

    assert (
        f(
            """
        if True:
            while False:
                # pragma: no cover
                a = 0

            b = 0
        else:
            c = 0  # pragma: no cover
    """
        )
        == {4, 5, 6, 9}
    )

    assert (
        f(
            """
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
    """
        )
        == {2, 3, 4, 6, 7, 8}
    )

    assert (
        f(
            """
        if TYPE_CHECKING:
            import cirq
            import foo
        def bar(a: 'cirq.Circuit'):
            pass
    """
        )
        == {2, 3, 4}
    )
