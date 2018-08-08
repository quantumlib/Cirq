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
import numpy as np
import pytest

import cirq


def test_unrelated():
    class Unrelated:
        pass

    cirq.testing.assert_unitary_effect_is(
        Unrelated(),
        expected_effect=None)

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Unrelated(),
            expected_effect=np.eye(2))


def test_has_unitary_effect():
    class Wrap:
        def __init__(self, has):
            self.has = has

        def _has_unitary_effect_(self):
            return self.has

    # When no unitary effect is present, only _has_unitary_effect_ is required.
    cirq.testing.assert_unitary_effect_is(
        Wrap(False),
        expected_effect=None)

    # Must provide the unitary effect if you say you have one.
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(True),
            expected_effect=np.eye(2))

    # Must agree with expectation.
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(False),
            expected_effect=np.eye(2))
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(True),
            expected_effect=None)


def test_maybe_unitary_effect():
    class Wrap:
        def __init__(self, mat):
            self.mat = mat

        def _maybe_unitary_effect_(self):
            return self.mat

    cirq.testing.assert_unitary_effect_is(
        Wrap(None),
        expected_effect=None)

    cirq.testing.assert_unitary_effect_is(
        Wrap(np.eye(2)),
        expected_effect=np.eye(2))

    cirq.testing.assert_unitary_effect_is(
        Wrap(np.diag([1, -1])),
        expected_effect=np.diag([1, -1]))

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(np.diag([1, -1])),
            expected_effect=np.eye(2))

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(None),
            expected_effect=np.eye(2))

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(np.diag([1, -1])),
            expected_effect=None)

    # Value error is not expected and so it escapes.
    class Fail:
        def _maybe_unitary_effect_(self):
            raise ValueError()
    with pytest.raises(ValueError):
        cirq.testing.assert_unitary_effect_is(
            Fail(),
            expected_effect=None)


def test_unitary_effect_fail():
    class FailValue:
        def _unitary_effect_(self):
            raise ValueError()

    class FailOther:
        def _unitary_effect_(self):
            raise NameError()

    cirq.testing.assert_unitary_effect_is(
        FailValue(),
        expected_effect=None)

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            FailValue(),
            expected_effect=np.eye(2))

    with pytest.raises(NameError):
        cirq.testing.assert_unitary_effect_is(
            FailOther(),
            expected_effect=None)


def test_unitary_effect_present():
    class Wrap:
        def __init__(self, mat):
            self.mat = mat

        def _unitary_effect_(self):
            return self.mat

    cirq.testing.assert_unitary_effect_is(
        Wrap(np.eye(2)),
        expected_effect=np.eye(2))

    cirq.testing.assert_unitary_effect_is(
        Wrap(np.diag([1, -1])),
        expected_effect=np.diag([1, -1]))

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(np.diag([1, -1])),
            expected_effect=np.eye(2))

    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(np.diag([1, -1])),
            expected_effect=None)

    # Returning None causes an assertion even if expected.
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(None),
            expected_effect=None)


def test_inconsistent():
    class Wrap:
        def __init__(self, has, get, may):
            self.has = has
            self.get = get
            self.may = may

        def _unitary_effect_(self):
            if self.get is ValueError:
                raise ValueError()
            return self.get

        def _maybe_unitary_effect_(self):
            return self.may

        def _has_unitary_effect_(self):
            return self.has

    cirq.testing.assert_unitary_effect_is(
        Wrap(False, ValueError, None),
        expected_effect=None)
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(True, ValueError, None),
            expected_effect=None)
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(False, np.eye(2), None),
            expected_effect=None)
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(False, ValueError, np.eye(2)),
            expected_effect=None)

    cirq.testing.assert_unitary_effect_is(
        Wrap(True, np.eye(2), np.eye(2)),
        expected_effect=np.eye(2))
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(False, np.eye(2), np.eye(2)),
            expected_effect=np.eye(2))
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(True, ValueError, np.eye(2)),
            expected_effect=np.eye(2))
    with pytest.raises(AssertionError):
        cirq.testing.assert_unitary_effect_is(
            Wrap(True, np.eye(2), None),
            expected_effect=np.eye(2))


def test_inherits():
    class Wrap(cirq.protocols.SupportsUnitaryEffect):
        def __init__(self, mat):
            self.mat = mat

        def _maybe_unitary_effect_(self):
            return self.mat

    cirq.testing.assert_unitary_effect_is(
        Wrap(None),
        expected_effect=None)

    cirq.testing.assert_unitary_effect_is(
        Wrap(np.eye(2)),
        expected_effect=np.eye(2))
