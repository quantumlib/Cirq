from typing import List, Mapping, Sequence

import collections
import time

import numpy as np
import matplotlib.pyplot as plt
import sympy

import cirq


class LivePlotCollector(cirq.Collector):
    """Performs concurrent collection of a parameter sweep over a circuit.

    Purposefully breaks down the sweep into individual calls, in order to test
    how samplers reacts to being given asynchronous sweeps to perform in
    parallel.

    Draws results live as they wait, start, and complete. (Note that this tends
    to slow things down quite a lot, so refreshes are limited to 5Hz.)
    """

    def __init__(
        self, *, circuit: cirq.Circuit, parameter: str, values: Sequence[float], repetitions: int
    ):
        self.next_index = 0
        self.circuit = circuit
        self.sweep = cirq.Points(parameter, values)
        self.resolvers = list(cirq.to_resolvers(self.sweep))
        self.reps = repetitions

        self.unstarted_xs = list(values)
        self.started_xs: List[float] = []
        self.result_xs: List[float] = []
        self.result_ys: Mapping[str, List[float]] = collections.defaultdict(list)

        self.fig = plt.figure()
        self.last_redraw_time = time.monotonic()

    def next_job(self):
        if self.next_index >= len(self.sweep):
            return None
        k = self.next_index
        p = self.sweep.points[k]
        self.next_index += 1
        self.started_xs.append(p)
        self.unstarted_xs.remove(p)
        return cirq.CircuitSampleJob(
            cirq.resolve_parameters(self.circuit, self.resolvers[k]), repetitions=self.reps, tag=p
        )

    def _redraw(self):
        self.fig.clear()
        plt.scatter(self.unstarted_xs, [0] * len(self.unstarted_xs), label='unstarted', s=1)
        plt.scatter(self.started_xs, [1] * len(self.started_xs), label='started', s=1)
        for k, v in self.result_ys.items():
            plt.scatter(self.result_xs, v, label=k, s=1)
        plt.xlabel(self.sweep.key)
        plt.legend()
        self.fig.canvas.draw()
        plt.pause(0.00001)

    def on_job_result(self, job, result):
        self.started_xs.remove(job.tag)
        self.result_xs.append(job.tag)
        for k, v in result.measurements.items():
            self.result_ys[k].append(np.mean(v.reshape(v.size)))

        t = time.monotonic()
        if t > self.last_redraw_time + 0.2 or not self.started_xs:
            self.last_redraw_time = t
            self._redraw()


def example():
    a, b = cirq.LineQubit.range(2)

    sampler = cirq.Simulator()

    circuit = cirq.Circuit(
        cirq.X(a) ** sympy.Symbol('t'),
        cirq.CNOT(a, b) ** sympy.Symbol('t'),
        cirq.measure(a, key='leader'),
        cirq.measure(b, key='follower'),
    )

    collector = LivePlotCollector(
        circuit=circuit, parameter='t', values=np.linspace(0, 1, 1000), repetitions=200
    )

    collector.collect(sampler, concurrency=5)
    plt.show()


if __name__ == '__main__':
    example()
