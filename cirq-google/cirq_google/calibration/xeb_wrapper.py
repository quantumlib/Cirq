import multiprocessing

import numpy as np

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.xeb_fitting as xebf
import cirq.experiments.xeb_sampling as xebsamp
from cirq_google.calibration.phased_fsim import (
    XEBPhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization, XEBPhasedFSimCalibrationOptions,
)


def run_calibration(
        calibration: XEBPhasedFSimCalibrationRequest,
        sampler: 'cirq.Sampler',
) -> PhasedFSimCalibrationResult:
    options: XEBPhasedFSimCalibrationOptions = calibration.options
    circuit = cirq.Circuit([calibration.gate.on(*pair) for pair in calibration.pairs])

    # 2. Set up XEB experiment
    cycle_depths = options.cycle_depths
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=options.n_library_circuits,
        two_qubit_gate=calibration.gate,
        max_cycle_depth=max(cycle_depths)
    )
    combs_by_layer = rqcg.get_random_combinations_for_layer_circuit(
        n_library_circuits=len(circuits),
        n_combinations=options.n_combinations,
        layer_circuit=circuit,
    )

    # 3. Sample data
    sampled_df = xebsamp.sample_2q_xeb_circuits(
        sampler=sampler,
        circuits=circuits,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs_by_layer,
    )

    # 4. Initial fidelities
    initial_fids = xebf.benchmark_2q_xeb_fidelities(
        sampled_df=sampled_df,
        circuits=circuits,
        cycle_depths=cycle_depths,
    )

    # 5. Characterize by fitting angles.
    pcircuits = [xebf.parameterize_circuit(circuit, options.fsim_options) for circuit in circuits]
    with multiprocessing.Pool() as pool:
        char_results = xebf.characterize_phased_fsim_parameters_with_xeb_by_pair(
            sampled_df=sampled_df,
            parameterized_circuits=pcircuits,
            cycle_depths=cycle_depths,
            options=options.fsim_options,
            pool=pool,
            fatol=options.fatol,
            xatol=options.xatol,
        )

    return PhasedFSimCalibrationResult(
        parameters={
            pair: PhasedFSimCharacterization(**param_dict)
            for pair, param_dict in char_results.final_params.items()
        },
        gate=calibration.gate,
        options=options,
        initial_fids=initial_fids,
        final_fids=char_results.fidelities_df,
    )
