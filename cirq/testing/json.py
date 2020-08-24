import io
import numpy as np
import pandas as pd

import cirq


def assert_json_roundtrip_works(obj, text_should_be=None, resolvers=None):
    """Tests that the given object can serialized and de-serialized

    Args:
        obj: The object to test round-tripping for.
        text_should_be: An optional argument to assert the JSON serialized
            output.
        resolvers: Any resolvers if testing those other than the default.

    Raises:
        AssertionError: The given object can not be round-tripped according to
            the given arguments.
    """
    buffer = io.StringIO()
    cirq.protocols.to_json(obj, buffer)

    if text_should_be is not None:
        buffer.seek(0)
        text = buffer.read()
        assert text == text_should_be, text

    buffer.seek(0)
    restored_obj = cirq.protocols.read_json(buffer, resolvers=resolvers)
    if isinstance(obj, np.ndarray):
        np.testing.assert_equal(restored_obj, obj)
    elif isinstance(obj, pd.DataFrame):
        pd.testing.assert_frame_equal(restored_obj, obj)
    elif isinstance(obj, pd.Index):
        pd.testing.assert_index_equal(restored_obj, obj)
    else:
        assert restored_obj == obj
