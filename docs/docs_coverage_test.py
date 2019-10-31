import inspect
from typing import Tuple, Type, Iterator, NamedTuple, Any, Dict, List

import cirq
from cirq._compat import seen_documentation, DocProperties

CirqMember = NamedTuple(
    'CirqMember',
    [
        ('scope', str),
        ('name', str),
        ('value', Any),
        ('kind', str),
        ('doc_props', DocProperties),
    ]
)


def _find_public_cirq_members() -> List[CirqMember]:
    module_scopes = [
        (cirq, 'cirq'),
        (cirq.experiments, 'cirq.experiments'),
        (cirq.google, 'cirq.google'),
        (cirq.testing, 'cirq.testing'),
    ]

    seen: Dict[str, CirqMember] = {}
    result: List[CirqMember] = []
    for module, scope in module_scopes:
        for name, obj in inspect.getmembers(module):
            if name in seen:
                if obj is not seen[name].value:
                    raise ValueError(f'Ambiguous name:\n'
                                     f'{seen[name].scope}.{name}\n'
                                     f'{scope}.{name}\n')
                continue
            if name.startswith('_') or inspect.ismodule(obj):
                continue

            if inspect.isclass(obj):
                kind = 'class'
            elif inspect.isfunction(obj):
                kind = 'function'
            else:
                kind = 'constant'

            doc_props = seen_documentation.get(name) or seen_documentation.get(id(obj))
            if doc_props is None:
                # coverage: ignore
                if kind == 'class':
                    example = f'''
                            from cirq._compat import documented

                            @documented(api_reference_category="gates")
                            class PublicClass:
                                ...
                        '''
                elif kind == 'function':
                    example = f'''
                            from cirq._compat import documented

                            @documented(api_reference_category="gates")
                            def public_function(...):
                                ...
                        '''
                else:
                    example = '''
                        from cirq._compat import documented

                        PUBLIC_CONSTANT = documented(\n'
                            value,\n'
                            """doc string

                            details
                            """,
                            api_reference_category="gates")
                    '''

                raise ValueError(
                    f'UNDOCUMENTED PUBLIC VALUE: {scope}.{name}\n'
                    f'\n'
                    f'Example of how to document:\n'
                    f'{example}'
                    f'\n'
                    f'UNDOCUMENTED PUBLIC VALUE: {scope}.{name}')

            item = CirqMember(value=obj,
                              scope=scope,
                              name=name,
                              kind=kind,
                              doc_props=doc_props)
            seen[name] = item
            result.append(item)
    return result


def test_public_values_equals_documented_values():
    members = _find_public_cirq_members()
    documented_names = {e for e in seen_documentation.keys() if isinstance(e, str)}
    seen_names = {e.name for e in members}
    private_docs = documented_names - seen_names
    assert not private_docs, (f"PRIVATE VALUES MARKED AS DOCUMENTED:\n"
                           f"{sorted(private_docs)}")
