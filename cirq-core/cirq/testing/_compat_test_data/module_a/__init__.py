# pylint: disable=wrong-or-nonexistent-copyright-notice
"""module_a for module deprecation tests"""

import logging

from cirq.testing._compat_test_data.module_a import module_b

from cirq.testing._compat_test_data.module_a.dupe import DUPE_CONSTANT as DUPE_CONSTANT

from cirq.testing._compat_test_data.module_a.types import SampleType as SampleType

MODULE_A_ATTRIBUTE = "module_a"

logging.info("init:module_a")
