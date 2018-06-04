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

"""Tests for results."""

import pytest

from cirq import study


class ResultMissingAll(study.TrialResult):
    pass


def test_result_missing_all():
    with pytest.raises(NotImplementedError):
        ResultMissingAll()


class ResultMissingParams(study.TrialResult):
    @property
    def repetitions(self):
        return None  # coverage: ignore

    @property
    def measurements(self):
        return None  # coverage: ignore


def test_result_missing_params():
    with pytest.raises(NotImplementedError):
        ResultMissingParams()


class ResultMissingRepetitions(study.TrialResult):
    @property
    def params(self):
        return None

    @property
    def measurements(self):
        return None  # coverage: ignore


def test_result_missing_repetitions():
    with pytest.raises(NotImplementedError):
        ResultMissingRepetitions()


class ResultMissingMeasurements(study.TrialResult):
    @property
    def params(self):
        return None

    @property
    def repetitions(self):
        return None


def test_result_missing_measurements():
    with pytest.raises(NotImplementedError):
        ResultMissingMeasurements()
