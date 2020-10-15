import cirq
import cirq.google.api.v2 as v2
import google.protobuf.text_format as text_format


def load_calibrations() -> cirq.google.Calibration:
    with open('metrics-pacific-2020-08-19.pbtxt') as f:
        metrics_str = f.read()
    metrics = text_format.Parse(metrics_str, v2.metrics_pb2.MetricsSnapshot())
    return cirq.google.Calibration(metrics)
