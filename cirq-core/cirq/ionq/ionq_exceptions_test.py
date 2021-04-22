# Copyright 2020 The Cirq Developers
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

import cirq.ionq as ionq


def test_ionq_exception():
    ex = ionq.IonQException(message='Hello', status_code=500)
    assert str(ex) == 'Status code: 500, Message: \'Hello\''
    assert ex.status_code == 500


def test_ionq_not_found_exception():
    ex = ionq.IonQNotFoundException(message='Where are you')
    assert str(ex) == 'Status code: 404, Message: \'Where are you\''
    assert ex.status_code == 404


def test_ionq_unsuccessful_job_exception():
    ex = ionq.IonQUnsuccessfulJobException(job_id='SWE', status='canceled')
    assert str(ex) == 'Status code: None, Message: \'Job SWE was canceled.\''
    assert ex.status_code is None
