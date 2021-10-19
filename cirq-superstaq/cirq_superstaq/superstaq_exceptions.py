# Copyright 2021 The Cirq Developers
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
"""Exceptions for the SuperstaQ API."""

import requests


class SuperstaQException(Exception):
    """An exception for errors coming from SuperstaQ's API.

    Attributes:
        status_code: A http status code, if coming from an http response with a failing status.
    """

    def __init__(self, message: str, status_code: int = None):
        super().__init__(f"Status code: {status_code}, Message: '{message}'")
        self.status_code = status_code


class SuperstaQModuleNotFoundException(SuperstaQException):
    """
    An exception for SuperstaQ features requiring an uninstalled module."""

    def __init__(self, name: str, context: str):
        message = f"'{context}' requires module '{name}'"
        super().__init__(message)


class SuperstaQNotFoundException(SuperstaQException):
    """An exception for errors from SuperstaQ's API when a resource is not found."""

    def __init__(self, message: str):
        super().__init__(message, status_code=requests.codes.not_found)


class SuperstaQUnsuccessfulJobException(SuperstaQException):
    """An exception for attempting to get info about an unsuccessful job.

    This exception occurs when a job has been canceled, deleted, or failed, and information about
    this job is attempted to be accessed.
    """

    def __init__(self, job_id: str, status: str):
        super().__init__(f"Job {job_id} was {status}.")
