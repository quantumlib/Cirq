import cirq
import sympy
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import cirq.google as cg

from cirq.work.observable_measurement import measure_observables_df, calibrate_readout_error
import cirq.contrib.noise_models as ccn

TRY_TO_USE_QUANTUM_ENGINE = True
OVERWRITE_CACHED_RESULTS = False


def get_sampler():
    if TRY_TO_USE_QUANTUM_ENGINE and 'GOOGLE_CLOUD_PROJECT' in os.environ:
        # coverage: ignore
        print("Using quantum engine")
        return cg.get_engine_sampler('rainbow', 'sqrt_iswap')

    print("Using noisy simulator")
    return cirq.DensityMatrixSimulator(noise=ccn.DepolarizingWithDampedReadoutNoiseModel(
        depol_prob=0.005, bitflip_prob=0.03, decay_prob=0.08))


def ansatz():
    a = sympy.Symbol('a')
    q = cirq.GridQubit(6, 5)
    qubits = [q]
    observables = [cirq.Z(q) * 1]
    circuit = cirq.Circuit([
        cirq.XPowGate(exponent=a).on(q)
    ])
    sweep = cirq.Linspace(a, 0.5, 2.5, 20)
    return circuit, observables, sweep, qubits


def simulate():
    circuit, observables, sweep, qubits = ansatz()
    simulator = cirq.Simulator()
    records = []

    print("Simulating observables ...")
    sim_results = simulator.simulate_sweep(circuit, sweep)
    for sim_result in sim_results:
        for term in observables:
            termval = term.expectation_from_wavefunction(
                sim_result.final_state,
                sim_result.qubit_map,
                check_preconditions=False)
            assert np.isclose(termval.imag, 0, atol=1e-4), termval.imag
            termval = termval.real

            records.append({
                'a': sim_result.params.value_of('a'),
                'mean': termval,
                'variance': 0,
            })

    results_df = pd.DataFrame(records)
    pd.to_pickle(results_df, 'measure-observables-readout-simulate.pickl')


def collect_data1():
    print("\nCollecting uncorrected data ...")
    circuit, observables, sweep, qubits = ansatz()
    results_df = measure_observables_df(
        circuit=circuit,
        observables=observables,
        sampler=get_sampler(),
        params=sweep,
        stopping_criteria='variance',
        stopping_criteria_val=1e-4,
        symmetrize_readout=False,
    )
    pd.to_pickle(results_df, 'measure-observables-readout-sample1.pickl')


def collect_data2():
    print("\nCollecting symmetrized data ...")
    circuit, observables, sweep, qubits = ansatz()
    results_df = measure_observables_df(
        circuit=circuit,
        observables=observables,
        sampler=get_sampler(),
        params=sweep,
        stopping_criteria='variance',
        stopping_criteria_val=1e-4,
        symmetrize_readout=True,
    )
    pd.to_pickle(results_df, 'measure-observables-readout-sample2.pickl')


def collect_data3():
    print("\nCalibrating readout error ...")
    circuit, observables, sweep, qubits = ansatz()
    readout_calibration = calibrate_readout_error(qubits=qubits,
                                                  sampler=get_sampler(),
                                                  stopping_criteria='variance',
                                                  stopping_criteria_val=1e-6)

    print("\nCollecting and correcting ...")
    results_df = measure_observables_df(
        circuit=circuit,
        observables=observables,
        sampler=get_sampler(),
        params=sweep,
        stopping_criteria='variance',
        stopping_criteria_val=1e-4,
        symmetrize_readout=True,
        readout_calibrations=readout_calibration
    )
    pd.to_pickle(results_df, 'measure-observables-readout-sample3.pickl')


def plot():
    simul_df = pd.read_pickle('measure-observables-readout-simulate.pickl')
    sample_df1 = pd.read_pickle('measure-observables-readout-sample1.pickl')
    sample_df2 = pd.read_pickle('measure-observables-readout-sample2.pickl')
    sample_df3 = pd.read_pickle('measure-observables-readout-sample3.pickl')

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 5))

    axl.axhline(0, color='grey')
    axl.plot(simul_df['a'], simul_df['mean'], '-', label='Noiseless')
    axl.plot(sample_df1['a'], sample_df1['mean'], '.-', label='Noisy')
    axl.plot(sample_df2['a'], sample_df2['mean'], '.-', label='Symmetrized')
    axl.plot(sample_df3['a'], sample_df3['mean'], '.-', label='Corrected')
    axl.legend(loc='best')

    simul_df = simul_df.set_index('a')
    sample_df1 = sample_df1.set_index('a')
    sample_df2 = sample_df2.set_index('a')
    sample_df3 = sample_df3.set_index('a')

    axr.axhline(0, color='grey')
    axr.plot([0.5], [0])  # advance the color cycle
    axr.errorbar(x=simul_df.index,
                 y=sample_df1['mean'] - simul_df['mean'],
                 yerr=sample_df1['variance'] ** 0.5,
                 capsize=5,
                 label='delta Noisy')
    axr.errorbar(x=simul_df.index,
                 y=sample_df2['mean'] - simul_df['mean'],
                 yerr=sample_df2['variance'] ** 0.5,
                 capsize=5,
                 label='delta Symm')
    axr.errorbar(x=simul_df.index,
                 y=sample_df3['mean'] - simul_df['mean'],
                 yerr=sample_df3['variance'] ** 0.5,
                 capsize=5,
                 label='delta corr')
    axr.legend(loc='best')

    axl.set_xlabel('a', fontsize=16)
    axr.set_xlabel('a', fontsize=16)
    axl.set_ylabel(r'$\langle Z \rangle$', fontsize=16)
    fig.tight_layout()
    fig.savefig('measure-observables-readout.png', dpi=200)


def main(cache=True):
    if not os.path.exists('measure-observables-readout-simulate.pickl') or not cache:
        simulate()
    if not os.path.exists('measure-observables-readout-sample1.pickl') or not cache:
        collect_data1()
    if not os.path.exists('measure-observables-readout-sample2.pickl') or not cache:
        collect_data2()
    if not os.path.exists('measure-observables-readout-sample3.pickl') or not cache:
        collect_data3()

    plot()


if __name__ == '__main__':
    main(cache=not OVERWRITE_CACHED_RESULTS)
