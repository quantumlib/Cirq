# Cirq version compatibility

## Overview
This document defines backwards compatibility guidelines across different versions of Cirq (either for code or data).

## Semantic Versioning 2.0
Cirq follows Semantic Versioning 2.0 ([semver](http://semver.org/)) for its public API. Each release version of Cirq has the form MAJOR.MINOR.PATCH. Changes to each number have the following meaning:

*   **MAJOR**: Potentially backwards incompatible changes. Code and data that worked with a previous major release will not necessarily work with the new release. However, in most cases existing json serialized Cirq objects would be migratable to the newer release; see [Compatibility of serialized objects](#compatibility-of-serialized-objects) for details on data compatibility.
*   **MINOR**: Backwards compatible features, speed improvements, etc. Code and data that worked with a previous minor release _and_ which depends only on the non-experimental public API will continue to work unchanged. For details on what is and is not the public API, see [What is covered](#what-is-covered).
*   **PATCH**: Backwards compatible bug fixes.

## What is covered
The public APIs of Cirq are backwards compatible across minor and patch versions. The public APIs consist of

* All the documented [symbols](https://quantumai.google/reference/python/cirq/all_symbols) (module, function, argument, property, class, or constant) in the Cirq module and its submodules, except for
    *   Private symbols: any symbol whose name start with an underscore **\_**.
    *   Experimental and `cirq.contrib` symbols, see [below](#what-is-not-covered) for details.
* Note that only code reachable through the `cirq` Python module is covered by the compatibility guarantees. 
    *   For example: Code in the [examples/](https://github.com/quantumlib/Cirq/tree/master/examples) and [dev\_tools/](https://github.com/quantumlib/Cirq/tree/master/dev_tools) directories is not reachable through the `cirq` Python module and is thus not covered by the compatibility guarantee. 
    *   Similarly, symbols in vendor packages, like [cirq-google](https://quantumai.google/reference/python/cirq_google/all_symbols), [cirq-aqt](https://quantumai.google/reference/python/cirq_aqt/all_symbols) are also not covered by the compatibility guarantee.
    *   If a symbol is available through the `cirq` Python module or its submodules, but is not documented, then it is **not** considered part of the public API.

## What is _not_ covered
* **Experimental APIs**: To facilitate development, we exempt some API symbols clearly marked as experimental from the compatibility guarantees. In particular, the following are not covered by any compatibility guarantees:
    *   any symbol in the `cirq.contrib` module or its submodules.
    *   any symbol whose name contains `experimental` or `Experimental`; or
    *   any symbol whose fully qualified name includes a module or class which is itself experimental. This includes fields and submessages of any json/protocol serialization called `experimental`.
* **Details of gate decompositions:** Many public gates and operations in Cirq are composite gates defined by specifying a particular decomposition in terms of other simpler gates, using `_decompose_` protocol. These decompositions may change for minor releases, with a guarantee that the old and new decompositions would result in the same unitary up to a global phase.
* **Floating point numerical details:** The specific floating point values computed by simulators / linalg utilities may change at any time. Users should rely only on approximate accuracy and numerical stability, not on the specific bits computed.
* **Type Preservation:** The exact type of the objects consumed and returned by methods / functions in the public API can be changed any time, as long as the change satisfies [Liskov Substitution Principle (LSP)](https://en.wikipedia.org/wiki/Liskov_substitution_principle).
* **Bugs:** We reserve the right to make backwards incompatible behavior (though not API) changes if the current implementation is clearly broken, that is, if it contradicts the documentation or if a well-known and well-defined intended behavior is not properly implemented due to a bug.
    *   For example: if a transformer claims to implement a well-known transformation algorithm but does not match that algorithm due to a bug, then we will fix the transformer. Our fix may break code relying on the wrong behavior of the transformer. We will note such changes in the release notes.
* **Circuit diagrams:** We reserve the right to change exact diagrams for gates / operations / circuits across minor releases. These changes would also be explicitly announced on cirq-announce@.
    *   For example: We can change [SVG diagrams](https://github.com/quantumlib/Cirq/issues/5689) and [text diagrams](https://github.com/quantumlib/Cirq/issues/5688) to use a different symbol for all classically controlled operations.
* **Error behavior:** We may replace input validation errors with non-error behavior. For instance, we may change a class/function to compute a result instead of raising an error for a certain set of inputs, even if that error is documented. We also reserve the right to change the text of error messages.
    *   For example: If we decide to extend functionality of `PauliSumExponential` to support non-commuting Pauli’s, the [value error](https://github.com/quantumlib/Cirq/blob/e00767a2ef1233e82e9089cf3801a77e4cc3aea3/cirq-core/cirq/ops/pauli_sum_exponential.py#L53) in the constructor would be removed.

## Compatibility of serialized objects
Cirq currently uses JSON serialization for serializing gates, operations and circuits. Many Cirq users serialize and store their circuits so they can be loaded and executed with a later release of Cirq. In compliance with [semver](https://semver.org/), serialized Json objects written with one version of Cirq can be loaded and evaluated with a later version of Cirq with the same major release.

We make additional guarantees for _supported_ Cirq serialized objects. A serialized object which was created using **only non-deprecated, non-experimental APIs** in Cirq major version `N` is a _supported serialization in version `N`_. Any serialized object supported in Cirq major version `N` can be loaded and executed with Cirq major version `N+1`. However, the functionality required to build or modify such an object may not be available anymore, so this guarantee only applies to the unmodified serialized objects.

We will endeavor to preserve backwards compatibility as long as possible, so that the serialized objects are usable over long periods of time.