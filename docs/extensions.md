## Extensions

The extension mechanism in cirq is designed to solve a specific kind of compatibility problem, which is best explained by an example.

Suppose library A defines a kind of thing called a "ReactiveIterable" and provides many utility methods for reactive iterables.
Separately, library B has implemented something called an "EventTrampoline".
It just so happens that an "EventTrampoline" is a "ReactiveIterable" in all but name.
If there was just some way to translate what library B provides into what library A wants, then one could apply all the great utility methods from library A on the value from library B.
Alas, there isn't.

The goal of the extension mechanism is to perform the translation between what B provides and what A wants.
Essentially, when library A was written, instead of having code like this:

```python
def some_method_in_A(reactive_iterable):
    ...
```

It would have code like this:

```python
def some_method_in_A(value, extensions):
    reactive_iterable = extensions.cast(value, ReactiveIterable)
    ...
```

And then, when you wanted library A to understand a type from library B, you would invoke the method like this:

```python
def your_code():
    ext = cirq.Extensions()
    ext.add_cast(desired_type=ReactiveIterable,
                 actual_type=EventTrampoline,
                 conversion=lambda trampoline: ...)

    some_method(trampoline, ext)
```

And EventTrampolines would be automatically converted into ReactiveIterables even though the people writing libraries A and B never knew about the other library.

### PotentialImplementation

An important part of the extension mechanism is the ability for classes to say *at runtime* whether or not they support a specific kind of functionality.
Classes that implement `PotentialImplementation` have a `try_cast(desired_type, extensions)` method that they use to tell a caller whether or not they support some desired functionality.

For example, the `RotZGate` can't have its effect scaled when the amount of rotation is a symbol instead of a number.
So it has a `try_cast` method that, when `desired_type` is set to `ExtrapolatableEffect`, returns `None` when the rotation angle is a symbol.
Otherwise it returns itself (indicating that its `matrix` method will not raise an error when called).

The `extensions` parameter given to `try_cast` is useful functionality in the callee depends on functionality being present in a dependency.
For example, `ControlledGate` only has a matrix if the gate it is controlling has a matrix.
