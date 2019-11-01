from cirq._compat import seen_documentation
from .api_generate import _all_public_cirq_members


def test_public_values_equals_documented_values():
    members = _all_public_cirq_members()
    documented_names = {
        e for e in seen_documentation.keys() if isinstance(e, str)
    }
    seen_names = {e.name for e in members}
    private_docs = documented_names - seen_names
    assert not private_docs, (f"PRIVATE VALUES MARKED AS DOCUMENTED:\n"
                              f"{sorted(private_docs)}")
