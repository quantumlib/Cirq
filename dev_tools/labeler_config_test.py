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


@pytest.mark.parametrize("label,globs", LABELER_RULES)
def test_labeler_globs_reference_existing_paths(label: str, globs: list[str]) -> None:
    del label
    for glob_pattern in globs:
        if _glob_has_non_prefix_wildcards(glob_pattern):
            continue
        prefix = _glob_prefix(glob_pattern)
        if prefix.startswith("**/"):
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
