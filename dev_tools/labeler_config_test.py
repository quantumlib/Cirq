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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELER_CONFIG = REPO_ROOT / ".github" / "labeler.yml"
LABELER_RULES = list(path_area_labeler.load_labeler_config(LABELER_CONFIG).items())


def test_labeler_config_is_non_empty() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    assert len(config) >= 20


@pytest.mark.parametrize("label,globs", LABELER_RULES)
def test_labeler_labels_use_supported_prefixes(label: str, globs: list[str]) -> None:
    del globs
    assert label.startswith("area/") or label.startswith("interface/")


def _glob_existence_prefix(glob_pattern: str) -> str | None:
    if _glob_has_non_prefix_wildcards(glob_pattern):
        return None
    prefix = _glob_prefix(glob_pattern)
    if prefix.startswith("**/"):
        return None
    return prefix


@pytest.mark.parametrize("label,globs", LABELER_RULES)
def test_labeler_globs_reference_existing_paths(label: str, globs: list[str]) -> None:
    del label
    for glob_pattern in globs:
        prefix = _glob_existence_prefix(glob_pattern)
        if prefix is None:
            continue
        assert (REPO_ROOT / prefix).exists(), f"Missing path prefix for glob {glob_pattern!r}"


def _glob_has_non_prefix_wildcards(glob_pattern: str) -> bool:
    remainder = glob_pattern.removeprefix("**/")
    return "*" in remainder.replace("/**", "")


def _glob_prefix(glob_pattern: str) -> str:
    if glob_pattern.endswith("/**"):
        return glob_pattern[:-3]
    if glob_pattern.startswith("**/"):
        return glob_pattern.removeprefix("**/").split("*", maxsplit=1)[0].rstrip("/")
    return glob_pattern.split("*", maxsplit=1)[0].rstrip("/")


def test_glob_has_non_prefix_wildcards_detects_middle_stars() -> None:
    assert _glob_has_non_prefix_wildcards(".github/workflows/release-*.yml")


def test_glob_prefix_for_recursive_and_directory_globs() -> None:
    assert _glob_prefix("**/setup.py") == "setup.py"
    assert _glob_prefix("docs/**") == "docs"
    assert _glob_prefix("cirq-core/setup.py") == "cirq-core/setup.py"


def test_glob_existence_prefix_skips_unanchored_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dev_tools.labeler_config_test._glob_prefix", lambda _glob_pattern: "**/docs"
    )
    assert _glob_existence_prefix("docs/**") is None
