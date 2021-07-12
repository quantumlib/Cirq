# Copyright 2021 The Cirq Developers
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

import os

import pytest

from dev_tools.requirements import explode


def _list_all_req_files():
    return [
        os.path.join(d, f)
        for d in ['dev_tools/requirements', 'dev_tools/requirements/deps']
        for f in os.listdir(d)
        if f.endswith(".txt")
    ]


@pytest.mark.parametrize('req', _list_all_req_files())
def test_validate_requirement_files(req):
    """Test that all req files are using valid requirements."""
    assert len(explode(req)) > 0


def test_explode():
    actual = explode("dev_tools/requirements/test_data/a.req.txt")
    assert actual == [
        'a_dependency==0.2.3',
        '# this one has spaces in front and at the end accidentally',
        '# a comment',
        'b',
        '# c is included',
        '# this one has no new line at the end',
        'c',
    ], actual
