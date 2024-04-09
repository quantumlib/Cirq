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
from pathlib import Path
from unittest import mock

import cirq_web


class FakeWidget(cirq_web.Widget):
    def __init__(self):
        super().__init__()

    def get_client_code(self) -> str:
        return "This is the test client code."

    def get_widget_bundle_name(self) -> str:
        return "testfile.txt"


def remove_whitespace(string: str) -> str:
    return "".join(string.split())


def test_repr_html(tmpdir):
    # # Reset the path so the files are accessible
    cirq_web.widget._DIST_PATH = Path(tmpdir) / "dir"
    path = tmpdir.mkdir('dir').join('testfile.txt')
    path.write("This is a test bundle")

    test_widget = FakeWidget()
    actual = test_widget._repr_html_()

    expected = f"""
        <meta charset="UTF-8">
        <div id="{test_widget.id}"></div>
        <script>This is a test bundle</script>
        This is the test client code.
        """

    assert remove_whitespace(expected) == remove_whitespace(actual)


@mock.patch.dict(os.environ, {"BROWSER": "true"})
def test_generate_html_file_with_browser(tmpdir):
    # # Reset the path so the files are accessible
    cirq_web.widget._DIST_PATH = Path(tmpdir) / "dir"
    path = tmpdir.mkdir('dir')

    testfile_path = path.join('testfile.txt')
    testfile_path.write("This is a test bundle")

    test_widget = FakeWidget()
    test_html_path = test_widget.generate_html_file(str(path), 'test.html', open_in_browser=True)
    actual = open(test_html_path, 'r', encoding='utf-8').read()

    expected = f"""
        <meta charset="UTF-8">
        <div id="{test_widget.id}"></div>
        <script>This is a test bundle</script>
        This is the test client code.
        """
    assert remove_whitespace(expected) == remove_whitespace(actual)
