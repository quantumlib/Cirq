<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.value

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Modules

[`abc_alt`](../cirq/value/abc_alt.md) module: A more flexible abstract base class metaclass ABCMetaImplementAnyOneOf.

[`angle`](../cirq/value/angle.md) module

[`digits`](../cirq/value/digits.md) module

[`duration`](../cirq/value/duration.md) module: A typed time delta that supports picosecond accuracy.

[`linear_dict`](../cirq/value/linear_dict.md) module: Linear combination represented as mapping of things to coefficients.

[`periodic_value`](../cirq/value/periodic_value.md) module

[`probability`](../cirq/value/probability.md) module: Utilities for handling probabilities.

[`random_state`](../cirq/value/random_state.md) module

[`timestamp`](../cirq/value/timestamp.md) module: A typed location in time that supports picosecond accuracy.

[`type_alias`](../cirq/value/type_alias.md) module

## Classes

[`class ABCMetaImplementAnyOneOf`](../cirq/value/ABCMetaImplementAnyOneOf.md): A metaclass extending `abc.ABCMeta` for defining abstract base classes

[`class Duration`](../cirq/value/Duration.md): A time delta that supports symbols and picosecond accuracy.

[`class LinearDict`](../cirq/value/LinearDict.md): Represents linear combination of things.

[`class PeriodicValue`](../cirq/value/PeriodicValue.md): Wrapper for periodic numerical values.

[`class Timestamp`](../cirq/value/Timestamp.md): A location in time with picosecond accuracy.

## Functions

[`alternative(...)`](../cirq/value/alternative.md): A decorator indicating an abstract method with an alternative default

[`big_endian_bits_to_int(...)`](../cirq/value/big_endian_bits_to_int.md): Returns the big-endian integer specified by the given bits.

[`big_endian_digits_to_int(...)`](../cirq/value/big_endian_digits_to_int.md): Returns the big-endian integer specified by the given digits and base.

[`big_endian_int_to_bits(...)`](../cirq/value/big_endian_int_to_bits.md): Returns the big-endian bits of an integer.

[`big_endian_int_to_digits(...)`](../cirq/value/big_endian_int_to_digits.md): Separates an integer into big-endian digits.

[`canonicalize_half_turns(...)`](../cirq/value/canonicalize_half_turns.md): Wraps the input into the range (-1, +1].

[`chosen_angle_to_canonical_half_turns(...)`](../cirq/value/chosen_angle_to_canonical_half_turns.md): Returns a canonicalized half_turns based on the given arguments.

[`chosen_angle_to_half_turns(...)`](../cirq/value/chosen_angle_to_half_turns.md): Returns a half_turns value based on the given arguments.

[`parse_random_state(...)`](../cirq/value/parse_random_state.md): Interpret an object as a pseudorandom number generator.

[`validate_probability(...)`](../cirq/value/validate_probability.md): Validates that a probability is between 0 and 1 inclusively.

[`value_equality(...)`](../cirq/value/value_equality.md): Implements __eq__/__ne__/__hash__ via a _value_equality_values_ method.

## Type Aliases

[`DURATION_LIKE`](../cirq/value/DURATION_LIKE.md)

[`Scalar`](../cirq/value/Scalar.md)

[`TParamVal`](../cirq/value/TParamVal.md)

