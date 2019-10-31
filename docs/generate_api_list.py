from collections import defaultdict

from docs.docs_coverage_test import _find_public_cirq_members


CATEGORY_DETAILS = {
    'Single Qubit Gates': 'Unitary operations you can apply to a single qubit.'
}

def main():
    categories = defaultdict(list)
    seen = set()
    for x in _find_public_cirq_members():
        if x.name in seen:
            continue
        seen.add(x.name)
        categories[x.doc_props.api_reference_category].append(x)

    print("""
.. currentmodule:: cirq

API Reference
=============

""")

    for category in sorted(categories.keys()):
        print(category)
        print("'" * len(category))
        print()
        print(CATEGORY_DETAILS.get(category) or '[SUMMARY]')
        print()
        print('.. autosummary::')
        print('    :toctree: generated/')
        print()
        for x in sorted(categories[category], key=lambda e: e.name):
            print('    ' + x.name)
        print()
        print()



if __name__ == '__main__':
    main()
