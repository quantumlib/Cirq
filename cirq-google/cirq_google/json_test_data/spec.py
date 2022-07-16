# pylint: disable=wrong-or-nonexistent-copyright-notice
import pathlib

import cirq_google
from cirq.testing.json import ModuleJsonTestSpec
from cirq_google.json_resolver_cache import _class_resolver_dictionary

TestSpec = ModuleJsonTestSpec(
    name="cirq_google",
    packages=[cirq_google, cirq_google.experimental],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[
        'Sycamore',
        'Sycamore23',
        'SQRT_ISWAP_INV_PARAMETERS',
        'ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION',
        'WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION',
    ],
    should_not_be_serialized=[
        'AnnealSequenceSearchStrategy',
        'CircuitOpDeserializer',
        'CircuitOpSerializer',
        'CircuitSerializer',
        'CIRCUIT_SERIALIZER',
        'CircuitWithCalibration',
        'Engine',
        'EngineJob',
        'EngineProcessor',
        'EngineProgram',
        'FSimPhaseCorrections',
        'NoiseModelFromGoogleNoiseProperties',
        'ProtoVersion',
        'GreedySequenceSearchStrategy',
        'PhasedFSimCalibrationError',
        'PhasedFSimEngineSimulator',
        'PerQubitDepolarizingWithDampedReadoutNoiseModel',
        'THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION',
        'ProcessorSampler',
        'ValidatingSampler',
        'CouldNotPlaceError',
        # Abstract:
        'ExecutableSpec',
    ],
    custom_class_name_to_cirq_type={
        k: f'cirq.google.{k}'
        for k in [
            'BitstringsMeasurement',
            'QuantumExecutable',
            'QuantumExecutableGroup',
            'KeyValueExecutableSpec',
            'ExecutableResult',
            'ExecutableGroupResult',
            'QuantumRuntimeConfiguration',
            'RuntimeInfo',
            'SharedRuntimeInfo',
            'ExecutableGroupResultFilesystemRecord',
            'NaiveQubitPlacer',
            'RandomDevicePlacer',
            'HardcodedQubitPlacer',
            'EngineProcessorRecord',
            'SimulatedProcessorRecord',
            'SimulatedProcessorWithLocalDeviceRecord',
            'EngineResult',
            'GridDevice',
            'GoogleCZTargetGateset',
        ]
    },
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)
