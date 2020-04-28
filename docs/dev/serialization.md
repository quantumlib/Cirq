# Serialization Guidelines

This developer document explains how Cirq serializes objects into (and out of)
JSON.
It also explains how to add a new serializable object,
how to remove a serializable object from cirq while maintaining backwards
compatibility with old serialized files, and various related guidelines.

## Exposed API

Most Cirq objects can be converted into json using the `cirq.to_json` method.
This is useful to users who want to keep track of the experiments that they have
run, or who want a simple way to communicate information between computers.

Here is an example of serializing an object:

```python
import cirq
import sympy

obj = cirq.X**sympy.Symbol('t')
text = cirq.to_json(obj)

print(text)
# prints:
# {
#   "cirq_type": "XPowGate",
#   "exponent": {
#     "cirq_type": "sympy.Symbol",
#     "name": "t"
#   },
#   "global_shift": 0.0
# }
```

The JSON can also be written to a file:

```
cirq.to_json(obj, filepath)
```

Or read back in from a file:

```
obj = cirq.read_json(filepath)
```

Or read back in from a string:

```python
deserialized_obj = cirq.read_json(json_text=text)
print(deserialized_obj)
# prints:
# X**t
```

## Mechanism

When writing JSON, Cirq checks if the given object has a `_json_dict_` method.
If it does, the object is replaced by the output of that method.
Otherwise, there are a series of several hardcoded cases for complex numbers,
numpy arrays, sympy expressions, and a few others.
The process of replacing an object by JSON proceeds recursively.
For example, the `_json_dict_` method may return a dictionary that contains a
value that is not JSON.
This value will be noticed, and converted into JSON using the same mechanism
(checking `_json_dict_`, checking hardcoded cases, etc).

When reading JSON, Cirq gives
[an object hook to json.loads](https://docs.python.org/3/library/json.html#encoders-and-decoders)
.
This hook checks if the object being parsed is a dictionary containing the key
"cirq_type".
If it is, Cirq looks up the associated value (the type string) in a hardcoded
dictionary in `cirq/protocols/json.py`.
That dictionary returns a callable object, usually a class, that maps the
dictionary into a parsed value.
If the returned object has a `_from_json_dict_` attribute, it is called instead.

## Adding a new serializable value

All of Cirq's public classes should be serializable.
This is enforced by the `test_json_test_data_coverage` test in
`cirq/protocols/json_serialization_test.py`, which iterates over cirq's API looking for types
with no associated json test data.

There are several steps needed to get a object serializing, deserializing, and
passing tests:

1. The object should have a `_json_dict_` method that returns a dictionary
containing a `"cirq_type"` key as well as keys for each of the value's
attributes.
If these keys do not match the names of the class' initializer arguments, a
`_from_json_dict_` class method must also be defined.
Typically the `"cirq_type"` will be the name of your class.

2. Add an entry to the big hardcoded dictionary in `cirq/protocols/json.py`,
mapping the cirq_type string you chose to the class.
You can also map the key to a helper method that returns the class (important
for backwards compatibility if e.g. a class is later replaced by another one).
After doing this, `cirq.to_json` and `cirq.read_json` should start working for
your object.

3. Add test data files to the `cirq/protocols/json_test_data` directory.
These are to ensure that the class remains deserializable in future versions.
There should be two files: `your_class_name.repr` and `your_class_name.json`.
`your_class_name.repr` should contain a python expression that evaluates to an
instances of your class, or a list of instances of your class.
The expression must eval correctly when only `cirq`, `pandas as pd`,
`numpy as np` and `sympy` have been imported.
Ideally, the contents of the `.repr` file are exactly the output of
`repr(your_obj)`.
`your_class_name.json` should contain the expected JSON output when serializing
the test value from `your_class_name.repr`.

## Removing a serializable value

When a serializable value is removed from cirq, old serialized instances
**must still work**.
They may deserialize to something different (but equivalent), but it is crucial
that they not fail to parse.
As such, "removing" a serializable value is more akin to removing it
*from the public API* as opposed to completely deleting it.

There are several steps:

1. Find the object's test files in `cirq/protocols/json_test_data`.
Change the file name extensions from `.json` to `.json_inward` and `.repr` to
`.repr_inward`.
This indicates that only deserialization needs to be tested, not deserialization
and serialization.
If `_inward` files already exist, merge into them (e.g. by ensuring they encode
lists and then appending into those lists).

2. Define a parsing method to stand in for the object.
This parsing method must return an object with the same basic behavior as the
object being removed, but does not have to return an exactly identical object.
For example, an X could be replaced by a PhasedX with no phasing.
Edit the entry in the big dictionary in `cirq/protocols/json.py` to point at
this method instead of the object being removed.
(There will likely be debate about exactly how to do this, on a case by case
basis.)
