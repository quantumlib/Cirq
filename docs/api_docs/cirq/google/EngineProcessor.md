<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.EngineProcessor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_reservation"/>
<meta itemprop="property" content="engine"/>
<meta itemprop="property" content="expected_down_time"/>
<meta itemprop="property" content="expected_recovery_time"/>
<meta itemprop="property" content="get_calibration"/>
<meta itemprop="property" content="get_current_calibration"/>
<meta itemprop="property" content="get_device"/>
<meta itemprop="property" content="get_device_specification"/>
<meta itemprop="property" content="get_reservation"/>
<meta itemprop="property" content="get_schedule"/>
<meta itemprop="property" content="health"/>
<meta itemprop="property" content="list_calibrations"/>
<meta itemprop="property" content="list_reservations"/>
<meta itemprop="property" content="remove_reservation"/>
<meta itemprop="property" content="supported_languages"/>
<meta itemprop="property" content="update_reservation"/>
</div>

# cirq.google.EngineProcessor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A processor available via the Quantum Engine API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.EngineProcessor`, `cirq.google.engine.engine.engine_processor.EngineProcessor`, `cirq.google.engine.engine_processor.EngineProcessor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.EngineProcessor(
    project_id: str,
    processor_id: str,
    context: "engine_base.EngineContext",
    _processor: Optional[qtypes.QuantumProcessor] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A project_id of the parent Google Cloud Project.
</td>
</tr><tr>
<td>
`processor_id`
</td>
<td>
Unique ID of the processor.
</td>
</tr><tr>
<td>
`context`
</td>
<td>
Engine configuration and context to use.
</td>
</tr><tr>
<td>
`_processor`
</td>
<td>
The optional current processor state.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A project_id of the parent Google Cloud Project.
</td>
</tr><tr>
<td>
`processor_id`
</td>
<td>
Unique ID of the processor.
</td>
</tr>
</table>



## Methods

<h3 id="create_reservation"><code>create_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_reservation(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    whitelisted_users: Optional[List[str]] = None
)
</code></pre>

Creates a reservation on this processor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start_time`
</td>
<td>
the starting date/time of the reservation.
</td>
</tr><tr>
<td>
`end_time`
</td>
<td>
the ending date/time of the reservation.
</td>
</tr><tr>
<td>
`whitelisted_users`
</td>
<td>
a list of emails that are allowed
to send programs during this reservation (in addition to users
with permission "quantum.reservations.use" on the project).
</td>
</tr>
</table>



<h3 id="engine"><code>engine</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>engine() -> "engine_base.Engine"
</code></pre>

Returns the parent Engine object.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The program's parent Engine.
</td>
</tr>

</table>



<h3 id="expected_down_time"><code>expected_down_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expected_down_time() -> "Optional[datetime.datetime]"
</code></pre>

Returns the start of the next expected down time of the processor, if
set.

<h3 id="expected_recovery_time"><code>expected_recovery_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expected_recovery_time() -> "Optional[datetime.datetime]"
</code></pre>

Returns the expected the processor should be available, if set.


<h3 id="get_calibration"><code>get_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_calibration(
    calibration_timestamp_seconds: int
) -> <a href="../../cirq/google/Calibration.md"><code>cirq.google.Calibration</code></a>
</code></pre>

Retrieve metadata about a specific calibration run.


#### Params:


* <b>`calibration_timestamp_seconds`</b>: The timestamp of the calibration in
    seconds since epoch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The calibration data.
</td>
</tr>

</table>



<h3 id="get_current_calibration"><code>get_current_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_current_calibration() -> Optional[<a href="../../cirq/google/Calibration.md"><code>cirq.google.Calibration</code></a>]
</code></pre>

Returns metadata about the current calibration for a processor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The calibration data or None if there is no current calibration.
</td>
</tr>

</table>



<h3 id="get_device"><code>get_device</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_device(
    gate_sets: Iterable[<a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a>]
) -> "cirq.Device"
</code></pre>

Returns a `Device` created from the processor's device specification.

This method queries the processor to retrieve the device specification,
which is then use to create a `SerializableDevice` that will validate
that operations are supported and use the correct qubits.

<h3 id="get_device_specification"><code>get_device_specification</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_device_specification() -> Optional[<a href="../../cirq/google/api/v2/device_pb2/DeviceSpecification.md"><code>cirq.google.api.v2.device_pb2.DeviceSpecification</code></a>]
</code></pre>

Returns a device specification proto for use in determining
information about the device.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Device specification proto if present.
</td>
</tr>

</table>



<h3 id="get_reservation"><code>get_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_reservation(
    reservation_id: str
)
</code></pre>

Retrieve a reservation given its id.


<h3 id="get_schedule"><code>get_schedule</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_schedule(
    from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
    to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2),
    time_slot_type: Optional[<a href="../../cirq/google/engine/client/quantum/enums/QuantumTimeSlot/TimeSlotType.md"><code>cirq.google.engine.client.quantum.enums.QuantumTimeSlot.TimeSlotType</code></a>] = None
) -> List[<a href="../../cirq/google/EngineTimeSlot.md"><code>cirq.google.EngineTimeSlot</code></a>]
</code></pre>

Retrieves the schedule for a processor.

The schedule may be filtered by time.

Time slot type will be supported in the future.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`from_time`
</td>
<td>
Filters the returned schedule to only include entries
that end no earlier than the given value. Specified either as an
absolute time (datetime.datetime) or as a time relative to now
(datetime.timedelta). Defaults to now (a relative time of 0).
Set to None to omit this filter.
</td>
</tr><tr>
<td>
`to_time`
</td>
<td>
Filters the returned schedule to only include entries
that start no later than the given value. Specified either as an
absolute time (datetime.datetime) or as a time relative to now
(datetime.timedelta). Defaults to two weeks from now (a relative
time of two weeks). Set to None to omit this filter.
</td>
</tr><tr>
<td>
`time_slot_type`
</td>
<td>
Filters the returned schedule to only include
entries with a given type (e.g. maintenance, open swim).
Defaults to None. Set to None to omit this filter.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Schedule time slots.
</td>
</tr>

</table>



<h3 id="health"><code>health</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>health() -> str
</code></pre>

Returns the current health of processor.


<h3 id="list_calibrations"><code>list_calibrations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_calibrations(
    earliest_timestamp_seconds: Optional[int] = None,
    latest_timestamp_seconds: Optional[int] = None
) -> List[<a href="../../cirq/google/Calibration.md"><code>cirq.google.Calibration</code></a>]
</code></pre>

Retrieve metadata about a specific calibration run.


#### Params:


* <b>`earliest_timestamp_seconds`</b>: The earliest timestamp of a calibration
    to return in UTC.
* <b>`latest_timestamp_seconds`</b>: The latest timestamp of a calibration to
    return in UTC.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The list of calibration data with the most recent first.
</td>
</tr>

</table>



<h3 id="list_reservations"><code>list_reservations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_reservations(
    from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
    to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2)
) -> List[<a href="../../cirq/google/EngineTimeSlot.md"><code>cirq.google.EngineTimeSlot</code></a>]
</code></pre>

Retrieves the reservations from a processor.

Only reservations from this processor and project will be
returned. The schedule may be filtered by starting and ending time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`from_time`
</td>
<td>
Filters the returned reservations to only include entries
that end no earlier than the given value. Specified either as an
absolute time (datetime.datetime) or as a time relative to now
(datetime.timedelta). Defaults to now (a relative time of 0).
Set to None to omit this filter.
</td>
</tr><tr>
<td>
`to_time`
</td>
<td>
Filters the returned reservations to only include entries
that start no later than the given value. Specified either as an
absolute time (datetime.datetime) or as a time relative to now
(datetime.timedelta). Defaults to two weeks from now (a relative
time of two weeks). Set to None to omit this filter.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of reservations.
</td>
</tr>

</table>



<h3 id="remove_reservation"><code>remove_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_reservation(
    reservation_id: str
)
</code></pre>




<h3 id="supported_languages"><code>supported_languages</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>supported_languages() -> List[str]
</code></pre>

Returns the list of processor supported program languages.


<h3 id="update_reservation"><code>update_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_processor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_reservation(
    reservation_id: str,
    start_time: datetime.datetime = None,
    end_time: datetime.datetime = None,
    whitelisted_users: List[str] = None
)
</code></pre>

Updates a reservation with new information.

Updates a reservation with a new start date, end date, or
list of additional users.  For each field, it the argument is left as
None, it will not be updated.



