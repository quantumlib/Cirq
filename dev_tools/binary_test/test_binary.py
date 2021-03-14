import pytest
import cirq.testing


def test_simple():
    def dead_code():
        # pylint: disable=unused-import
        import cirq.testing._compat_test_data
        from cirq.testing._compat_test_data import fake_a

    print("hello")
    with cirq.testing.assert_deprecated("import", deadline="v0.20"):
        dead_code()

    from cirq.google.api import v1, v2

    assert v2.program_pb2.Circuit()
    assert v1.params_pb2.ParameterSweep()


if __name__ == '__main__':
    pytest.main([__file__])
