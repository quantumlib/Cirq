# Copyright 2024 The Cirq Developers
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

import multiprocessing
import os
import pathlib
from collections.abc import Iterator
from typing import Any, Hashable

import pytest

import cirq
from cirq.protocols.json_serialization_test import MODULE_TEST_SPECS


REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent

_EXCLUDE_JSON_FILES = (
    # sympy - related objects
    "cirq-core/cirq/protocols/json_test_data/sympy.Add.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.E.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Equality.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.EulerGamma.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Float.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.GreaterThan.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Integer.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.LessThan.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Mul.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Pow.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Rational.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.StrictGreaterThan.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.StrictLessThan.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Symbol.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.Unequality.json",
    "cirq-core/cirq/protocols/json_test_data/sympy.pi.json",
    # RigettiQCSAspenDevice does not pickle
    "cirq-rigetti/cirq_rigetti/json_test_data/RigettiQCSAspenDevice.json",
    # TODO(#6674,pavoljuhas) - fix pickling of ProjectorSum
    "cirq-core/cirq/protocols/json_test_data/ProjectorSum.json",
)


def _is_included(json_filename: str) -> bool:
    if any(json_filename.endswith(t) for t in _EXCLUDE_JSON_FILES):
        return False
    if not os.path.isfile(json_filename):
        return False
    # exclude list objects
    with open(json_filename) as fp:
        if fp.read(8).startswith("["):
            return False
    return True


@pytest.fixture(scope='module')
def pool() -> Iterator[multiprocessing.pool.Pool]:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        yield pool


def _read_json_on_worker(json_filename: str) -> Any:
    return cirq.read_json(json_filename)


def test_exclude_json_files_has_valid_entries() -> None:
    """Verify _EXCLUDE_JSON_FILES has valid entries."""
    invalid_json_paths = [f for f in _EXCLUDE_JSON_FILES if not REPO_ROOT.joinpath(f).is_file()]
    assert invalid_json_paths == []


@pytest.mark.xfail
@pytest.mark.parametrize(
    'json_filename',
    [
        f"{abs_path}.json"
        for m in MODULE_TEST_SPECS
        for abs_path in m.all_test_data_keys()
        if _is_included(f"{abs_path}.json")
    ],
)
def test_hash_from_pickle(json_filename: str, pool: multiprocessing.pool.Pool):
    obj_local = cirq.read_json(json_filename)
    if not isinstance(obj_local, Hashable):
        return
    obj_worker = pool.apply(_read_json_on_worker, [json_filename])
    assert obj_worker == obj_local
    assert hash(obj_worker) == hash(obj_local)
