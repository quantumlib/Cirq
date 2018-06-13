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

    assert f("a = 0  # coverage: ignore") == {1}

    assert f("""
        a = 0  # coverage: ignore
        b = 0
    """) == {2}

    assert f("""
        a = 0  
        b = 0  # coverage: ignore
    """) == {3}

    assert f("""
        a = 0  # coverage: ignore  
        b = 0  # coverage: ignore
    """) == {2, 3}

    assert f("""
        if True:
            a = 0  # coverage: ignore

            b = 0
    """) == {3}

    assert f("""
        if True:
            # coverage: ignore
            a = 0

            b = 0
    """) == {3, 4, 5, 6, 7}

    assert f("""
        if True:
            # coverage: ignore
            a = 0

            b = 0
        stop = 1
    """) == {3, 4, 5, 6}

    assert f("""
        if True:
            # coverage: ignore
            a = 0

            b = 0
        else:
            c = 0
    """) == {3, 4, 5, 6}

    assert f("""
        if True:
            while False:
                # coverage: ignore
                a = 0

            b = 0
        else:
            c = 0  # coverage: ignore
    """) == {4, 5, 6, 9}

    assert f("""
        a = 2#coverage:ignore
        a = 3 #coverage:ignore
        a = 4# coverage:ignore
        a = 5#coverage :ignore
        a = 6#coverage: ignore
        a = 7#coverage: ignore\t
        a = 8#coverage:\tignore\t
        
        b = 1 # no cover
        b = 2 # coverage: definitely
        b = 3 # lint: ignore
    """) == {2, 3, 4, 5, 6, 7, 8}
