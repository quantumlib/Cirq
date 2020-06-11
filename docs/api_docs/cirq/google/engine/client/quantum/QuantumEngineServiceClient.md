<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine.client.quantum.QuantumEngineServiceClient" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="cancel_quantum_job"/>
<meta itemprop="property" content="cancel_quantum_reservation"/>
<meta itemprop="property" content="create_quantum_job"/>
<meta itemprop="property" content="create_quantum_program"/>
<meta itemprop="property" content="create_quantum_reservation"/>
<meta itemprop="property" content="delete_quantum_job"/>
<meta itemprop="property" content="delete_quantum_program"/>
<meta itemprop="property" content="delete_quantum_reservation"/>
<meta itemprop="property" content="from_service_account_file"/>
<meta itemprop="property" content="from_service_account_json"/>
<meta itemprop="property" content="get_quantum_calibration"/>
<meta itemprop="property" content="get_quantum_job"/>
<meta itemprop="property" content="get_quantum_processor"/>
<meta itemprop="property" content="get_quantum_program"/>
<meta itemprop="property" content="get_quantum_reservation"/>
<meta itemprop="property" content="get_quantum_result"/>
<meta itemprop="property" content="list_quantum_calibrations"/>
<meta itemprop="property" content="list_quantum_job_events"/>
<meta itemprop="property" content="list_quantum_jobs"/>
<meta itemprop="property" content="list_quantum_processors"/>
<meta itemprop="property" content="list_quantum_programs"/>
<meta itemprop="property" content="list_quantum_reservation_budgets"/>
<meta itemprop="property" content="list_quantum_reservation_grants"/>
<meta itemprop="property" content="list_quantum_reservations"/>
<meta itemprop="property" content="list_quantum_time_slots"/>
<meta itemprop="property" content="quantum_run_stream"/>
<meta itemprop="property" content="reallocate_quantum_reservation_grant"/>
<meta itemprop="property" content="update_quantum_job"/>
<meta itemprop="property" content="update_quantum_program"/>
<meta itemprop="property" content="update_quantum_reservation"/>
<meta itemprop="property" content="SERVICE_ADDRESS"/>
</div>

# cirq.google.engine.client.quantum.QuantumEngineServiceClient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



-

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.client.quantum_v1alpha1.QuantumEngineServiceClient`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.engine.client.quantum.QuantumEngineServiceClient(
    transport: QUANTUM_ENGINE_SERVICE_GRPC_TRANSPORT_LIKE = None,
    channel: Optional[grpc.Channel] = None,
    credentials: Optional[service_account.Credentials] = None,
    client_config: Optional[Dict[str, Any]] = None,
    client_info: Optional[google.api_core.gapic_v1.client_info.ClientInfo] = None,
    client_options: Union[Dict[str, Any], google.api_core.client_options.ClientOptions] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
transport (Union[~.QuantumEngineServiceGrpcTransport,
Callable[[~.Credentials, type], ~.QuantumEngineServiceGrpcTransport]): A transport
instance, responsible for actually making the API calls.
The default transport uses the gRPC protocol.
This argument may also be a callable which returns a
transport instance. Callables will be sent the credentials
as the first argument and the default transport class as
the second argument.
channel (grpc.Channel): DEPRECATED. A ``Channel`` instance
through which to make calls. This argument is mutually exclusive
with ``credentials``; providing both will raise an exception.
credentials (google.auth.credentials.Credentials): The
authorization credentials to attach to requests. These
credentials identify this application to the service. If none
are specified, the client will attempt to ascertain the
credentials from the environment.
This argument is mutually exclusive with providing a
transport instance to ``transport``; doing so will raise
an exception.
client_config (dict): DEPRECATED. A dictionary of call options for
each method. If not specified, the default configuration is used.
client_info (google.api_core.gapic_v1.client_info.ClientInfo):
The client info used to send a user-agent string along with
API requests. If ``None``, then default info will be used.
Generally, you only need to set this if you're developing
your own client library.
client_options (Union[dict, google.api_core.client_options.ClientOptions]):
Client options used to set user options on the client. API Endpoint
should be set through client_options.
</td>
</tr>

</table>



## Methods

<h3 id="cancel_quantum_job"><code>cancel_quantum_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cancel_quantum_job(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> client.cancel_quantum_job()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="cancel_quantum_reservation"><code>cancel_quantum_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cancel_quantum_reservation(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.cancel_quantum_reservation()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="create_quantum_job"><code>create_quantum_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_quantum_job(
    parent: Optional[str] = None,
    quantum_job: Union[Dict[str, Any], pb_types.QuantumJob] = None,
    overwrite_existing_run_context: Optional[bool] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.create_quantum_job()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
quantum_job (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob`
overwrite_existing_run_context (bool): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="create_quantum_program"><code>create_quantum_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_quantum_program(
    parent: Optional[str] = None,
    quantum_program: Union[Dict[str, Any], pb_types.QuantumProgram] = None,
    overwrite_existing_source_code: Optional[bool] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.create_quantum_program()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
quantum_program (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram`
overwrite_existing_source_code (bool): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="create_quantum_reservation"><code>create_quantum_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_quantum_reservation(
    parent: Optional[str] = None,
    quantum_reservation: Union[Dict[str, Any], pb_types.QuantumReservation] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.create_quantum_reservation()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
quantum_reservation (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="delete_quantum_job"><code>delete_quantum_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_quantum_job(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> client.delete_quantum_job()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="delete_quantum_program"><code>delete_quantum_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_quantum_program(
    name: Optional[str] = None,
    delete_jobs: Optional[bool] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> client.delete_quantum_program()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
delete_jobs (bool): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="delete_quantum_reservation"><code>delete_quantum_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_quantum_reservation(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> client.delete_quantum_reservation()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="from_service_account_file"><code>from_service_account_file</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_service_account_file(
    filename, *args, **kwargs
)
</code></pre>

Creates an instance of this client using the provided credentials
file.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
filename (str): The path to the service account private key json
file.
</td>
</tr>
<tr>
<td>
`args`
</td>
<td>
Additional arguments to pass to the constructor.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Additional arguments to pass to the constructor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`QuantumEngineServiceClient`
</td>
<td>
The constructed client.
</td>
</tr>
</table>



<h3 id="from_service_account_json"><code>from_service_account_json</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_service_account_json(
    filename, *args, **kwargs
)
</code></pre>

Creates an instance of this client using the provided credentials
file.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
filename (str): The path to the service account private key json
file.
</td>
</tr>
<tr>
<td>
`args`
</td>
<td>
Additional arguments to pass to the constructor.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Additional arguments to pass to the constructor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`QuantumEngineServiceClient`
</td>
<td>
The constructed client.
</td>
</tr>
</table>



<h3 id="get_quantum_calibration"><code>get_quantum_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_calibration(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_calibration()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="get_quantum_job"><code>get_quantum_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_job(
    name: Optional[str] = None,
    return_run_context: Optional[bool] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_job()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
return_run_context (bool): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="get_quantum_processor"><code>get_quantum_processor</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_processor(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_processor()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="get_quantum_program"><code>get_quantum_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_program(
    name: Optional[str] = None,
    return_code: Optional[bool] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_program()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
return_code (bool): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="get_quantum_reservation"><code>get_quantum_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_reservation(
    name: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_reservation()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="get_quantum_result"><code>get_quantum_result</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_quantum_result(
    parent: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.get_quantum_result()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumResult` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_calibrations"><code>list_quantum_calibrations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_calibrations(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_calibrations():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_calibrations().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_job_events"><code>list_quantum_job_events</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_job_events(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_job_events():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_job_events().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJobEvent` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_jobs"><code>list_quantum_jobs</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_jobs(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_jobs():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_jobs().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_processors"><code>list_quantum_processors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_processors(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_processors():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_processors().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_programs"><code>list_quantum_programs</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_programs(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_programs():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_programs().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_reservation_budgets"><code>list_quantum_reservation_budgets</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_reservation_budgets(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_reservation_budgets():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_reservation_budgets().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservationBudget` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_reservation_grants"><code>list_quantum_reservation_grants</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_reservation_grants(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_reservation_grants():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_reservation_grants().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservationGrant` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_reservations"><code>list_quantum_reservations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_reservations(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_reservations():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_reservations().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="list_quantum_time_slots"><code>list_quantum_time_slots</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_quantum_time_slots(
    parent: Optional[str] = None,
    page_size: Optional[int] = None,
    filter_: Optional[str] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> # Iterate over all results
>>> for element in client.list_quantum_time_slots():
...     # process element
...     pass
>>>
>>>
>>> # Alternatively:
>>>
>>> # Iterate over results one page at a time
>>> for page in client.list_quantum_time_slots().pages:
...     for element in page:
...         # process element
...         pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
parent (str): -
page_size (int): The maximum number of resources contained in the
underlying API response. If page streaming is performed per-
resource, this parameter does not affect the return value. If page
streaming is performed per-page, this determines the maximum number
of resources in a page.
filter_ (str): -
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~google.api_core.page_iterator.PageIterator` instance.
An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumTimeSlot` instances.
You can also iterate over the pages of the response
using its `pages` property.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="quantum_run_stream"><code>quantum_run_stream</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>quantum_run_stream(
    requests,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> request = {}
>>>
>>> requests = [request]
>>> for element in client.quantum_run_stream(requests):
...     # process element
...     pass



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
requests (iterator[dict|cirq.google.engine.client.quantum_v1alpha1.proto.engine_pb2.QuantumRunStreamRequest]): The input objects. If a dict is provided, it must be of the
same form as the protobuf message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumRunStreamRequest`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Iterable[~cirq.google.engine.client.quantum_v1alpha1.types.QuantumRunStreamResponse].
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="reallocate_quantum_reservation_grant"><code>reallocate_quantum_reservation_grant</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reallocate_quantum_reservation_grant(
    name: Optional[str] = None,
    source_project_id: Optional[str] = None,
    target_project_id: Optional[str] = None,
    duration: Union[Dict[str, Any], duration_pb2.Duration] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.reallocate_quantum_reservation_grant()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
source_project_id (str): -
target_project_id (str): -
duration (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.Duration]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.Duration`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservationGrant` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="update_quantum_job"><code>update_quantum_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_quantum_job(
    name: Optional[str] = None,
    quantum_job: Union[Dict[str, Any], pb_types.QuantumJob] = None,
    update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.update_quantum_job()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
quantum_job (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob`
update_mask (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="update_quantum_program"><code>update_quantum_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_quantum_program(
    name: Optional[str] = None,
    quantum_program: Union[Dict[str, Any], pb_types.QuantumProgram] = None,
    update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.update_quantum_program()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
quantum_program (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram`
update_mask (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>



<h3 id="update_quantum_reservation"><code>update_quantum_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/client/quantum_v1alpha1/gapic/quantum_engine_service_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_quantum_reservation(
    name: Optional[str] = None,
    quantum_reservation: Union[Dict[str, Any], pb_types.QuantumReservation] = None,
    update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
    retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
    timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
    metadata: Optional[Sequence[Tuple[str, str]]] = None
)
</code></pre>

-


#### Example:

>>> from cirq.google.engine.client import quantum_v1alpha1
>>>
>>> client = quantum_v1alpha1.QuantumEngineServiceClient()
>>>
>>> response = client.update_quantum_reservation()



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
name (str): -
quantum_reservation (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation`
update_mask (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask]): -

If a dict is provided, it must be of the same form as the protobuf
message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask`
retry (Optional[google.api_core.retry.Retry]):  A retry object used
to retry requests. If ``None`` is specified, requests will
be retried using a default configuration.
timeout (Optional[float]): The amount of time, in seconds, to wait
for the request to complete. Note that if ``retry`` is
specified, the timeout applies to each individual attempt.
metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
that is provided to the method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`google.api_core.exceptions.GoogleAPICallError`
</td>
<td>
If the request
failed for any reason.
</td>
</tr><tr>
<td>
`google.api_core.exceptions.RetryError`
</td>
<td>
If the request failed due
to a retryable error and retry attempts failed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the parameters are invalid.
</td>
</tr>
</table>





## Class Variables

* `SERVICE_ADDRESS = 'quantum.googleapis.com:443'` <a id="SERVICE_ADDRESS"></a>
