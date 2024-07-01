# Copyright 2020 The Cirq Developers
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
"""Exceptions for the IonQ API."""

from typing import Optional

import requests


class IonQException(Exception):
    """An exception for errors coming from IonQ's API.

    Attributes:
        status_code: A http status code, if coming from an http response with a failing status.
    """

    def __init__(self, message, status_code: Optional[int] = None):
        super().__init__(f'Status code: {status_code}, Message: \'{message}\'')
        self.status_code = status_code


class IonQNotFoundException(IonQException):
    """An exception for errors from IonQ's API when a resource is not found."""

    def __init__(self, message):
        super().__init__(message, status_code=requests.codes.not_found)


class IonQUnsuccessfulJobException(IonQException):
    """An exception for attempting to get info about an unsuccessful job.

    This exception occurs when a job has been canceled, deleted, or failed, and information about
    this job is attempted to be accessed.
    """

    def __init__(self, job_id: str, status: str):
        super().__init__(f'Job {job_id} was {status}.')


class IonQSerializerMixedGatesetsException(Exception):
    """An exception for IonQ serializer when attempting to run a batch of circuits
    that do not have the same type of gates (either 'qis' or 'native' gates).
    """

    def __init__(self, message):
        super().__init__(f'Message: \'{message}\'')
