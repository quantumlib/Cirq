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

import pytest

from dev_tools import path_area_labeler

LABELER_CONFIG = pathlib.Path(".github/labeler.yml")


def test_load_labeler_config_parses_rules() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    assert config["area/gates"] == ["cirq-core/cirq/ops/**"]
    assert "cirq-google/cirq_google/engine/**" in config["area/google/engine"]


@pytest.mark.parametrize(
    "glob_pattern,path,expected",
    [
        ("cirq-core/cirq/ops/**", "cirq-core/cirq/ops/common_gates.py", True),
        ("cirq-core/cirq/ops/**", "cirq-core/cirq/sim/sparse_simulator.py", False),
        ("**/setup.py", "cirq-core/setup.py", True),
        ("docs/**", "docs/dev/triage.md", True),
    ],
)
def test_glob_matches(glob_pattern: str, path: str, expected: bool) -> None:
    assert path_area_labeler.glob_matches(glob_pattern, path) is expected


def test_labels_for_paths_applies_multiple_matches() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    labels = path_area_labeler.labels_for_paths(
        config,
        [
            "cirq-core/cirq/ops/common_gates.py",
            "cirq-google/cirq_google/engine/engine.py",
        ],
    )
    assert labels == {"area/gates", "area/google", "area/google/engine", "interface/cirq-google"}


def test_extract_paths_from_text_finds_backtick_and_plain_paths() -> None:
    text = """
    Bug in `cirq-core/cirq/sim/sparse_simulator.py` when calling gates.
    Also see docs/dev/triage.md for context.
    """
    paths = path_area_labeler.extract_paths_from_text(text)
    assert paths == {
        "cirq-core/cirq/sim/sparse_simulator.py",
        "docs/dev/triage.md",
    }


def test_labels_for_issue_text() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    text = "Failure in cirq-ionq/cirq_ionq/service.py when submitting jobs."
    paths = path_area_labeler.extract_paths_from_text(text)
    labels = path_area_labeler.labels_for_paths(config, paths)
    assert labels == {"interface/cirq-ionq"}
