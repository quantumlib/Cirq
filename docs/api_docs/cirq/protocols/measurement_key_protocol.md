<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.measurement_key_protocol" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RaiseTypeErrorIfNotProvided"/>
</div>

# Module: cirq.protocols.measurement_key_protocol

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/measurement_key_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Protocol for object that have measurement keys.



## Classes

[`class SupportsMeasurementKey`](../../cirq/protocols/SupportsMeasurementKey.md): An object that is a measurement and has a measurement key or keys.

## Functions

[`is_measurement(...)`](../../cirq/protocols/is_measurement.md): Determines whether or not the given value is a measurement.

[`measurement_key(...)`](../../cirq/protocols/measurement_key.md): Get the single measurement key for the given value.

[`measurement_keys(...)`](../../cirq/protocols/measurement_keys.md): Gets the measurement keys of measurements within the given value.

## Other Members

* `RaiseTypeErrorIfNotProvided` <a id="RaiseTypeErrorIfNotProvided"></a>
