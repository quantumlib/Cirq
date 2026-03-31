# Style guidelines

As mentioned in [CONTRIBUTING.md](https://github.com/quantumlib/Cirq/blob/main/CONTRIBUTING.md) we use use [Pylint](https://pylint.pycqa.org/)
to check for style violations.  Pylint attempts to enforce styles in
[PEP 8](https://www.python.org/dev/peps/pep-0008/). To see which lint checks we enforce,
see the `[tool.pylint.messages_control]` section in the
[pyproject.toml](https://github.com/quantumlib/Cirq/blob/main/pyproject.toml) file.

Here we include some extra style guidelines.

## Import statements

We follow the [import standards](https://www.python.org/dev/peps/pep-0008/#imports) of PEP 8,
with the following guidance.

### Module import path conventions

We use two different conventions for Python `import` statements, depending on whether the `import`
statement is in main implementation code or test code.

#### Normal code

In Cirq's main implementation code (not testing code), we prefer importing the full module. This
aids in mocking during tests.  Thus we prefer
```python
from cirq import ops
qubit = ops.NamedQubit('a')
```
in contrast to
```python
from cirq.ops import NamedQubit
qubit = NamedQubit('a')
```
or (the one we would prefer, but doing this causes cyclic dependencies)
```python
import cirq
qubit = cirq.NamedQubit('a')
```
The one exception to this is for the typing code, where we prefer the direct import
```python
from typing import Mapping
```
This exception allows typing hints to be more compact.

#### Test code

In tests, however, we prefer that we use Cirq as you would use Cirq externally. For code
that is in the Cirq core framework, this is
```python
import cirq
qubit = cirq.NamedQubit('a')
```
For Cirq code that is outside of the core and does not appear at the `cirq` module level,
for example work in `contrib`, one should use the highest level possible for test code
```python
import cirq
from cirq import contrib
contrib.circuit_to_latex_using_qcircuit(cirq.Circuit())
```

Of course, if this import style fundamentally cannot be used, do not let this
block submitting a pull request for the code as we will definitely grant
exceptions.

### Ordering of module import statements

The import statements are alphabetically ordered in 3 groups for standard Python modules,
third-party modules, and for internal Cirq imports.  This ordering is enforced by the CI.
In a local development environment, the import statements can be sorted either by using
the [isort](https://pycqa.github.io/isort/) program or by running the
`check/format-incremental` script.

## Type annotations

Cirq makes extensive use of type annotations as defined by
[PEP 484](https://peps.python.org/pep-0484/). All new code should use type
annotations where possible, especially on public classes and functions to serve
as documentation, but also on internal code so that the mypy type checker can
help catch coding errors.

For documentation purposes in particular, type annotations should match the way
classes and functions are accessed by Cirq users, which is typically on the
top-level `cirq` namespace (for example, users use `cirq.Sampler` even though
the sampler class is defined in `cirq.work.sampler.Sampler`). Code in `cirq-core`
typically cannot import and use `cirq.Sampler` directly because this could
create an import cycle where modules import each other (perhaps indirectly).
Instead, the import of the top-level `cirq` library can be guarded by the
`TYPE_CHECKING` constant provided by `typing`. In addition, the module needs to
start with `from __future__ import annotations` so that annotations are skipped
at the module import time and get only evaluated during type-checking when the
`cirq` import is in effect.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq

def accepts_sampler(sampler: cirq.Sampler) -> None:
    ...
```

Use top-level `cirq.*` annotations like this when annotating new public types
and classes. Older code may not adhere to this style and should be updated.

Note that type annotations may need to be quoted when they act in expressions
that are evaluated at the import time, for example,

```python
from typing import Union

MOMENT_OR_OPERATION = Union['cirq.Moment', 'cirq.Operation']
```

## Nomenclature

Using consistent wording across Cirq is important for lowering users'
cognitive load. For rule governing naming, see the
[nomenclature guidelines](nomenclature.md).

## Datetimes

Prefer using timezone-aware `datetime` objects.

```python
import datetime
dt = datetime.datetime.now(tz=datetime.timezone.utc)
```

Public components of Protobuf APIs will return "aware" `datetime` objects.
JSON de-serialization will promote values to "aware" `datetime` objects upon deserialization.

Comparing (or testing equality) between "naive" and "aware" `datetime` objects throws
an exception.
If you are implementing a class that has `datetime` member variables, delegate equality
and comparison operators to the built-in `datetime` equality and comparison operators.
If you're writing a function that compares `datetime` objects, you can defensively promote
them to "aware" objects or use their `.timestamp()` properties for a comparison that will
never throw an exception.

Absolutely do not use `datetime.utcnow()` as explained in the warnings in the
Python [documentation](https://docs.python.org/3/library/datetime.html).
