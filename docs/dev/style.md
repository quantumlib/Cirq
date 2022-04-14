# Style guidelines

As mentioned in [CONTRIBUTING.md](https://github.com/quantumlib/Cirq/blob/master/CONTRIBUTING.md) we use use [pylint](https://www.pylint.org/) 
to check for style violations.  Pylint attempts to enforce styles in 
[PEP 8](https://www.python.org/dev/peps/pep-0008/). To see which lint checks we enforce, see the 
[dev_tools/conf/.pylintrc](https://github.com/quantumlib/Cirq/blob/master/dev_tools/conf/.pylintrc) file.

Here we include some extra style guidelines.

## Import statements

We follow the [import standards](https://www.python.org/dev/peps/pep-0008/#imports) of PEP 8, 
with the following guidance.  

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
from typing import List
```
This exception allows typing hints to be more compact. 

In tests, however, we prefer that we use Cirq as you would use cirq externally. For code
that is in the Cirq core framework this is
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

## Type annotations

Cirq makes extensive use of type annotations as defined by
[PEP 484](https://peps.python.org/pep-0484/). All new code should use type
annotations where possible, especially on public classes and functions to serve
as documentation, but also on internal code so that the mypy typechecker can
help catch coding errors.

For documentation purposes in particular, type annotations should match the way
classes and functions are accessed by cirq users, which is typically on the
top-level `cirq` namespace (for example, users use `cirq.Sampler` even though
the sampler class is defined in `cirq.work.sampler.Sampler`). Code in cirq-core
typically cannot import and use `cirq.Sampler` directly because this could
create an import cycle where modules import each other (perhaps indirectly).
Instead, the import of the top-level `cirq` library can be guarded by the
`TYPE_CHECKING` constant provided by `typing`, and the type annotation can be
quoted so that it is not evaluated when the module is imported but rather during
type-checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq

def accepts_sampler(sampler: 'cirq.Sampler') -> None:
    ...
```

Use top-level `cirq.*` annotations like this when annotating new public types
and classes. Older code may not adhere to this style and should be updated.

Note that type annotations may need to be quoted like this in other situations
as well, such as when an annotation is a "forward reference" to a class defined
later in the same file.

## Nomenclature

Using consistent wording across Cirq is important for lowering users
cognitive load. For rule governing naming, see the 
[nomenclature guidelines](nomenclature.md).
