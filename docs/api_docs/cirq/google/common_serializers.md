<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.common_serializers" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CZ_POW_DESERIALIZER"/>
<meta itemprop="property" content="CZ_POW_SERIALIZER"/>
<meta itemprop="property" content="CZ_SERIALIZER"/>
<meta itemprop="property" content="LIMITED_FSIM_DESERIALIZER"/>
<meta itemprop="property" content="LIMITED_FSIM_SERIALIZERS"/>
<meta itemprop="property" content="MEASUREMENT_DESERIALIZER"/>
<meta itemprop="property" content="MEASUREMENT_SERIALIZER"/>
<meta itemprop="property" content="PHYSICAL_Z"/>
<meta itemprop="property" content="SINGLE_QUBIT_DESERIALIZERS"/>
<meta itemprop="property" content="SINGLE_QUBIT_HALF_PI_DESERIALIZERS"/>
<meta itemprop="property" content="SINGLE_QUBIT_HALF_PI_SERIALIZERS"/>
<meta itemprop="property" content="SINGLE_QUBIT_SERIALIZERS"/>
<meta itemprop="property" content="SQRT_ISWAP_DESERIALIZERS"/>
<meta itemprop="property" content="SQRT_ISWAP_SERIALIZERS"/>
<meta itemprop="property" content="SYC_DESERIALIZER"/>
<meta itemprop="property" content="SYC_SERIALIZER"/>
<meta itemprop="property" content="VIRTUAL_Z"/>
<meta itemprop="property" content="WAIT_GATE_DESERIALIZER"/>
<meta itemprop="property" content="WAIT_GATE_SERIALIZER"/>
</div>

# Module: cirq.google.common_serializers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/common_serializers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Common Serializers that can be used by APIs.


This file contains the following serializers (and corresponding deserializers)

    - SINGLE_QUBIT_SERIALIZERS: A list of GateOpSerializer for single qubit
        rotations using cirq Gates.
    - MEASUREMENT_SERIALIZER:  Single GateOpSerializer for the measurement gate
    - SINGLE_QUBIT_SERIALIZERS: A list of GateOpSerializer for single qubit
        rotations confined to half-pi increments using cirq Gates.

## Other Members

* `CZ_POW_DESERIALIZER` <a id="CZ_POW_DESERIALIZER"></a>
* `CZ_POW_SERIALIZER` <a id="CZ_POW_SERIALIZER"></a>
* `CZ_SERIALIZER` <a id="CZ_SERIALIZER"></a>
* `LIMITED_FSIM_DESERIALIZER` <a id="LIMITED_FSIM_DESERIALIZER"></a>
* `LIMITED_FSIM_SERIALIZERS` <a id="LIMITED_FSIM_SERIALIZERS"></a>
* `MEASUREMENT_DESERIALIZER` <a id="MEASUREMENT_DESERIALIZER"></a>
* `MEASUREMENT_SERIALIZER` <a id="MEASUREMENT_SERIALIZER"></a>
* `PHYSICAL_Z = 'physical'` <a id="PHYSICAL_Z"></a>
* `SINGLE_QUBIT_DESERIALIZERS` <a id="SINGLE_QUBIT_DESERIALIZERS"></a>
* `SINGLE_QUBIT_HALF_PI_DESERIALIZERS` <a id="SINGLE_QUBIT_HALF_PI_DESERIALIZERS"></a>
* `SINGLE_QUBIT_HALF_PI_SERIALIZERS` <a id="SINGLE_QUBIT_HALF_PI_SERIALIZERS"></a>
* `SINGLE_QUBIT_SERIALIZERS` <a id="SINGLE_QUBIT_SERIALIZERS"></a>
* `SQRT_ISWAP_DESERIALIZERS` <a id="SQRT_ISWAP_DESERIALIZERS"></a>
* `SQRT_ISWAP_SERIALIZERS` <a id="SQRT_ISWAP_SERIALIZERS"></a>
* `SYC_DESERIALIZER` <a id="SYC_DESERIALIZER"></a>
* `SYC_SERIALIZER` <a id="SYC_SERIALIZER"></a>
* `VIRTUAL_Z = 'virtual_propagates_forward'` <a id="VIRTUAL_Z"></a>
* `WAIT_GATE_DESERIALIZER` <a id="WAIT_GATE_DESERIALIZER"></a>
* `WAIT_GATE_SERIALIZER` <a id="WAIT_GATE_SERIALIZER"></a>
