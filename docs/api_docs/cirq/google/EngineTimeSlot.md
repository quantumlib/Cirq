<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.EngineTimeSlot" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="to_proto"/>
<meta itemprop="property" content="maintenance_description"/>
<meta itemprop="property" content="maintenance_title"/>
<meta itemprop="property" content="project_id"/>
<meta itemprop="property" content="slot_type"/>
</div>

# cirq.google.EngineTimeSlot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_timeslot.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A python wrapping of a Quantum Engine timeslot.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.EngineTimeSlot`, `cirq.google.engine.engine_timeslot.EngineTimeSlot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.EngineTimeSlot(
    processor_id, start_time, end_time,
    slot_type=<TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED: 0>, project_id=None,
    maintenance_title=None, maintenance_description=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`processor_id`
</td>
<td>
The processor whose schedule the time slot exists on.
</td>
</tr><tr>
<td>
`start_time`
</td>
<td>
starting datetime of the time slot, usually in local time.
</td>
</tr><tr>
<td>
`end_time`
</td>
<td>
ending datetime of the time slot, usually in local time.
</td>
</tr><tr>
<td>
`slot_type`
</td>
<td>
type of time slot (reservation, open swim, etc)
</td>
</tr><tr>
<td>
`project_id`
</td>
<td>
Google Cloud Platform id of the project, as a string
</td>
</tr><tr>
<td>
`maintenance_title`
</td>
<td>
If a MAINTENANCE period, a string title describing the
type of maintenance being done.
</td>
</tr><tr>
<td>
`maintenance_description`
</td>
<td>
If a MAINTENANCE period, a string describing the
particulars of the maintenancethe title of the slot
</td>
</tr>
</table>



## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_timeslot.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_proto(
    proto: qtypes.QuantumTimeSlot
)
</code></pre>




<h3 id="to_proto"><code>to_proto</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_timeslot.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_proto()
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






## Class Variables

* `maintenance_description = None` <a id="maintenance_description"></a>
* `maintenance_title = None` <a id="maintenance_title"></a>
* `project_id = None` <a id="project_id"></a>
* `slot_type = <TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED: 0>` <a id="slot_type"></a>
