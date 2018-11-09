`typed_ast` is a Python 3 package that provides a Python 2.7 and Python 3
parser similar to the standard `ast` library.  Unlike `ast`, the parsers in
`typed_ast` include PEP 484 type comments and are independent of the version of
Python under which they are run.  The `typed_ast` parsers produce the standard
Python AST (plus type comments), and are both fast and correct, as they are
based on the CPython 2.7 and 3.6 parsers.

