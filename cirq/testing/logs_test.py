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

import logging
import warnings

import pytest

import cirq.testing


def test_assert_logs_valid_single_logs():
    with cirq.testing.assert_logs('apple'):
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', 'orange'):
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs():
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', 'fruit'):
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple') as logs:
        logging.error('orange apple fruit')
    assert len(logs) == 1
    assert logs[0].getMessage() == 'orange apple fruit'
    assert logs[0].levelno == logging.ERROR

    with cirq.testing.assert_logs('apple'):
        warnings.warn('orange apple fruit')


def test_assert_logs_invalid_single_logs():
    match = ('^dog expected to appear in log messages but it was not found. '
             'Logs messages: \\[\'orange apple fruit\'\\].$')
    with pytest.raises(AssertionError, match=match):
        with cirq.testing.assert_logs('dog'):
            logging.error('orange apple fruit')

    with pytest.raises(AssertionError, match='dog'):
        with cirq.testing.assert_logs('dog', 'cat'):
            logging.error('orange apple fruit')


def test_assert_logs_valid_multiple_logs():
    with cirq.testing.assert_logs('apple', count=2):
        logging.error('orange apple fruit')
        logging.error('other')

    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', 'other', count=2):
        logging.error('other')
        logging.error('orange apple fruit')

    with cirq.testing.assert_logs('apple', count=3):
        logging.error('orange apple fruit')
        logging.error('other')
        logging.warning('other two')


def test_assert_logs_invalid_multiple_logs():
    match = '^Expected 1 log message but got 2.$'
    with pytest.raises(AssertionError,
                       match='^Expected 1 log message but got 2.$'):
        with cirq.testing.assert_logs('dog'):
            logging.error('orange apple fruit')
            logging.error('dog')

    with pytest.raises(AssertionError,
                       match='^Expected 2 log message but got 3.$'):
        with cirq.testing.assert_logs('dog', count=2):
            logging.error('orange apple fruit')
            logging.error('other')
            logging.error('dog')

    match = ('^dog expected to appear in log messages but it was not found. '
             'Logs messages: \\[\'orange\', \'other\', \'whatever\'\\].$')
    with pytest.raises(AssertionError, match=match):
        with cirq.testing.assert_logs('dog', count=3):
            logging.error('orange')
            logging.error('other')
            logging.error('whatever')


def test_assert_logs_log_level():
    # Default is warning
    with cirq.testing.assert_logs('apple'):
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')
    with cirq.testing.assert_logs('apple', 'critical', count=2):
        logging.critical('critical')
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')
    with cirq.testing.assert_logs('apple', level=logging.INFO, count=2):
        logging.error('orange apple fruit')
        logging.debug('should not')
        logging.info('count')


def test_assert_logs_warnings():
    # Capture all warnings in one context, so that test cases that will
    # display a warning do not do so when the test is run.
    with warnings.catch_warnings(record=True):
        with cirq.testing.assert_logs('apple'):
            warnings.warn('orange apple fruit')

        with cirq.testing.assert_logs('apple', count=2):
            warnings.warn('orange apple fruit')
            logging.error('other')

        with cirq.testing.assert_logs('apple', capture_warnings=False):
            logging.error('orange apple fruit')
            warnings.warn('warn')

        with pytest.raises(AssertionError,
                           match='^Expected 1 log message but got 0.$'):
            with cirq.testing.assert_logs('apple', capture_warnings=False):
                warnings.warn('orange apple fruit')
