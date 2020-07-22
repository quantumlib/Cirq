from pprint import pprint

import scipy.interpolate

import cirq.contrib.noise_models as ccn

import cirq
import sympy
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import cirq.google as cg

from cirq.work.observable_measurement import measure_observables_df

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


def hydrogen_jw_hamiltonian():
    # Generated from openfermion
    q0, q1, q2, q3 = cirq.GridQubit.rect(1, 4, 5, 1)
    terms = [
        (0.1711977489805745 + 0j) * cirq.Z(q0),
        (0.17119774898057447 + 0j) * cirq.Z(q1),
        (-0.2227859302428765 + 0j) * cirq.Z(q2),
        (-0.22278593024287646 + 0j) * cirq.Z(q3),
        (0.16862219157249939 + 0j) * cirq.Z(q0) * cirq.Z(q1),
        (0.04532220205777764 + 0j) * cirq.Y(q0) * cirq.X(q1) * cirq.X(q2) * cirq.Y(q3),
        (-0.04532220205777764 + 0j) * cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2) * cirq.X(q3),
        (-0.04532220205777764 + 0j) * cirq.X(q0) * cirq.X(q1) * cirq.Y(q2) * cirq.Y(q3),
        (0.04532220205777764 + 0j) * cirq.X(q0) * cirq.Y(q1) * cirq.Y(q2) * cirq.X(q3),
        (0.12054482203290037 + 0j) * cirq.Z(q0) * cirq.Z(q2),
        (0.16586702409067802 + 0j) * cirq.Z(q0) * cirq.Z(q3),
        (0.16586702409067802 + 0j) * cirq.Z(q1) * cirq.Z(q2),
        (0.12054482203290037 + 0j) * cirq.Z(q1) * cirq.Z(q3),
        (0.1743484418396392 + 0j) * cirq.Z(q2) * cirq.Z(q3)
    ]
    return terms, [q0, q1, q2, q3]


def ansatz(quick=False):
    """Create a mildly interesting circuit with two parameters (arbitrary)."""
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    observables, (q0, q1, q2, q3) = hydrogen_jw_hamiltonian()
    qubits = [q0, q1, q2, q3]
    circuit = cirq.Circuit([
        cirq.YPowGate(exponent=0.5).on_each(*qubits),
        cirq.ISWAP(q0, q1) ** 0.5,
        cirq.ISWAP(q2, q3) ** 0.5,
        cirq.XPowGate(exponent=b).on_each(q1, q2),
        cirq.ISWAP(q1, q2) ** 0.5,
        cirq.XPowGate(exponent=a).on_each(q0, q3),
    ])

    res = 2 if quick else 12
    sweep = cirq.Product(
        cirq.Linspace(a, 0, 1, res),
        cirq.Linspace(b, 0, 1, res),
    )
    return circuit, observables, sweep, qubits


def simulate(quick=False):
    circuit, observables, sweep, qubits = ansatz(quick=quick)
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
                'b': sim_result.params.value_of('b'),
                'mean': termval,
                'variance': 0,
            })

    results_df = pd.DataFrame(records)
    print(results_df)
    agg_df = results_df.groupby(['a', 'b']).sum()
    print(agg_df)
    agg_df = agg_df.reset_index()
    pd.to_pickle(agg_df, 'measure-observables-vqe-simulate.pickl')


def collect_data(quick=False):
    circuit, observables, sweep, qubits = ansatz(quick=quick)
    print("Circuit:")
    print(circuit)
    print("\nObservables:")
    pprint(observables)
    print("\nSweep:")
    print(sweep)

    results_df = measure_observables_df(
        circuit=circuit,
        observables=observables,
        sampler=get_sampler(),
        params=sweep,
        stopping_criteria='repetitions',
        stopping_criteria_val=20_000,
    )
    print()
    print(results_df)
    agg_df = results_df.groupby(['a', 'b']).sum()
    print(agg_df)
    agg_df = agg_df.reset_index()
    pd.to_pickle(agg_df, 'measure-observables-vqe-sample.pickl')


def interpolate_for_plot(df, im_res=200):
    xx, yy = np.meshgrid(np.linspace(0, 1, im_res), np.linspace(0, 1, im_res))
    zz = scipy.interpolate.griddata(
        points=df[['a', 'b']].values,
        values=df['mean'].values,
        xi=(xx, yy),
        method='nearest',
    )
    return zz


def plot():
    simul_df = pd.read_pickle('measure-observables-vqe-simulate.pickl')
    sample_df = pd.read_pickle('measure-observables-vqe-sample.pickl')

    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(10, 5))
    axl, axr = ImageGrid(fig=fig,
                         rect=111,
                         nrows_ncols=(1, 2),
                         cbar_mode='single',
                         axes_pad=0.15,
                         cbar_pad=0.15,
                         )

    axl.set_title('Simulated', fontsize=16)
    simul_zz = interpolate_for_plot(simul_df)
    norm = plt.Normalize(simul_zz.min(), simul_zz.max())
    im = axl.imshow(simul_zz,
                    extent=(0, 1, 0, 1),
                    norm=norm,
                    origin='lower', cmap='PuOr',
                    interpolation='none')
    axr.set_title('Sampled', fontsize=16)
    im = axr.imshow(interpolate_for_plot(sample_df),
                    extent=(0, 1, 0, 1),
                    norm=norm,
                    origin='lower', cmap='PuOr',
                    interpolation='none')

    axl.set_xlabel('a', fontsize=16)
    axr.set_xlabel('a', fontsize=16)
    axl.set_ylabel('b', fontsize=16)
    axr.cax.colorbar(im)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig('measure-observables-vqe.png', dpi=200)


def main(cache=True, quick=False):
    if not os.path.exists('measure-observables-vqe-simulate.pickl') or not cache:
        simulate(quick=quick)
    if not os.path.exists('measure-observables-vqe-sample.pickl') or not cache:
        collect_data(quick=quick)

    plot()


if __name__ == '__main__':
    main(cache=not OVERWRITE_CACHED_RESULTS, quick=False)
