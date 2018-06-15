## Extensions

The experimental extension mechanism in cirq is designed to solve a specific kind of compatibility problem, which is best explained by an example.

Suppose library A defines a kind of thing called a "ReactiveIterable" and provides many utility methods for reactive iterables.
Separately, library B has implemented something called an "EventTrampoline".
It just so happens that an "EventTrampoline" is a "ReactiveIterable" in all but name.
If there was just some way to translate what library B provides into what library A wants, then one could apply all the great utility methods from library A on the value from library B.
Alas, there isn't.

The goal of the extension mechanism is to perform the translatation between what B provides and what A wants.
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
