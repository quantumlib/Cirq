import cirq
import cirq_google
import datetime
import pandas as pd
import sympy
import numpy as np

import glob
import os


## END IMPORTS


def get_LocalXEBPhasedFSimCalibrationOptions():
    return cirq_google.LocalXEBPhasedFSimCalibrationOptions(
        n_library_circuits=20,
        n_combinations=10,
        cycle_depths=(5, 25, 50, 100, 200, 300),
        fatol=0.005,
        xatol=0.005,
        fsim_options=cirq.experiments.XEBPhasedFSimCharacterizationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=True,
            characterize_gamma=True,
            characterize_phi=True,
            theta_default=0.0,
            zeta_default=0.0,
            chi_default=0.0,
            gamma_default=0.0,
            phi_default=0.0,
        ),
        n_processes=None,
    )


def get_FloquetPhasedFSimCalibrationRequest():
    return cirq_google.FloquetPhasedFSimCalibrationRequest(
        gate=cirq.FSimGate(theta=0.7853981633974483, phi=0.0),
        pairs=(
            (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            (cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
        ),
        options=cirq_google.FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
            readout_error_tolerance=0.4,
        ),
    )


def get_CouplerPulse():
    return cirq_google.experimental.CouplerPulse(
        hold_time=cirq.Duration(nanos=10),
        coupling_mhz=25.0,
        rise_time=cirq.Duration(nanos=18),
        padding_time=cirq.Duration(picos=2500.0),
    )


def get_FloquetPhasedFSimCalibrationOptions():
    return cirq_google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=True,
        characterize_zeta=True,
        characterize_chi=False,
        characterize_gamma=True,
        characterize_phi=False,
        readout_error_tolerance=0.4,
    )


def get_GateTabulation():
    return cirq_google.optimizers.two_qubit_gates.gate_compilation.GateTabulation(
        np.array(
            [
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (6.123233995736766e-17 + 0j), -1j, 0j],
                [0j, -1j, (6.123233995736766e-17 + 0j), 0j],
                [0j, 0j, 0j, (0.8660254037844387 - 0.49999999999999994j)],
            ],
            dtype=np.complex128,
        ),
        np.array(
            [
                [0.7853981633974483, 0.7853981633974483, 0.1308996938995748],
                [0.5455546451806152, 0.4274250500660468, 1.1102230246251565e-16],
                [0.482263980185593, 0.15541244226118112, -0.0],
                [0.08994436657892213, 0.0688663429298958, -0.0],
                [0.5337019863142523, 0.31800967733278185, -0.2565859360625904],
                [0.48438278888761155, 0.39121774851710633, 0.28728266960007875],
            ],
            dtype=np.float64,
        ),
        [
            [],
            [
                (
                    np.array(
                        [
                            [
                                (-0.8113361037323902 + 0.388228151720065j),
                                (0.38855300559639544 - 0.20009795309891054j),
                            ],
                            [
                                (-0.38855300559639544 - 0.20009795309891054j),
                                (-0.8113361037323902 - 0.388228151720065j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (-0.15500592487333173 - 0.5209557609876208j),
                                (-0.6259723589563565 + 0.5592288119996914j),
                            ],
                            [
                                (0.6259723589563565 + 0.5592288119996914j),
                                (-0.15500592487333173 + 0.5209557609876208j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                )
            ],
            [
                (
                    np.array(
                        [
                            [
                                (-0.7591335658137299 - 0.5262321684934362j),
                                (0.37132129473675946 + 0.09442685090928131j),
                            ],
                            [
                                (-0.37132129473675946 + 0.09442685090928131j),
                                (-0.7591335658137299 + 0.5262321684934362j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (0.004722472920167423 + 0.9809944588837683j),
                                (0.18002103957975657 + 0.07224953423714474j),
                            ],
                            [
                                (-0.18002103957975657 + 0.07224953423714474j),
                                (0.004722472920167423 - 0.9809944588837683j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                )
            ],
            [
                (
                    np.array(
                        [
                            [
                                (-0.0897362442086916 + 0.02445105905520244j),
                                (-0.8503296766825128 - 0.5179662084918382j),
                            ],
                            [
                                (0.8503296766825128 - 0.5179662084918382j),
                                (-0.0897362442086916 - 0.02445105905520244j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (0.7441127742234152 + 0.6642425418602642j),
                                (0.009545063892332065 + 0.07061810374004159j),
                            ],
                            [
                                (-0.009545063892332065 + 0.07061810374004159j),
                                (0.7441127742234152 - 0.6642425418602642j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                )
            ],
            [
                (
                    np.array(
                        [
                            [
                                (-0.8113361037323902 + 0.388228151720065j),
                                (0.38855300559639544 - 0.20009795309891054j),
                            ],
                            [
                                (-0.38855300559639544 - 0.20009795309891054j),
                                (-0.8113361037323902 - 0.388228151720065j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (-0.15500592487333173 - 0.5209557609876208j),
                                (-0.6259723589563565 + 0.5592288119996914j),
                            ],
                            [
                                (0.6259723589563565 + 0.5592288119996914j),
                                (-0.15500592487333173 + 0.5209557609876208j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                ),
                (
                    np.array(
                        [
                            [
                                (-0.8113361037323902 + 0.388228151720065j),
                                (0.38855300559639544 - 0.20009795309891054j),
                            ],
                            [
                                (-0.38855300559639544 - 0.20009795309891054j),
                                (-0.8113361037323902 - 0.388228151720065j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (-0.15500592487333173 - 0.5209557609876208j),
                                (-0.6259723589563565 + 0.5592288119996914j),
                            ],
                            [
                                (0.6259723589563565 + 0.5592288119996914j),
                                (-0.15500592487333173 + 0.5209557609876208j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                ),
            ],
            [
                (
                    np.array(
                        [
                            [
                                (0.41295648139511504 - 0.17280776715854063j),
                                (0.5430881450957462 - 0.7103940362502399j),
                            ],
                            [
                                (-0.5430881450957462 - 0.7103940362502399j),
                                (0.41295648139511504 + 0.17280776715854063j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (0.19013479709876246 + 0.4941656851052965j),
                                (-0.6469746262664201 + 0.5487010730480224j),
                            ],
                            [
                                (0.6469746262664201 + 0.5487010730480224j),
                                (0.19013479709876246 - 0.4941656851052965j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                ),
                (
                    np.array(
                        [
                            [
                                (0.41295648139511504 - 0.17280776715854063j),
                                (0.5430881450957462 - 0.7103940362502399j),
                            ],
                            [
                                (-0.5430881450957462 - 0.7103940362502399j),
                                (0.41295648139511504 + 0.17280776715854063j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                    np.array(
                        [
                            [
                                (0.19013479709876246 + 0.4941656851052965j),
                                (-0.6469746262664201 + 0.5487010730480224j),
                            ],
                            [
                                (0.6469746262664201 + 0.5487010730480224j),
                                (0.19013479709876246 - 0.4941656851052965j),
                            ],
                        ],
                        dtype=np.complex128,
                    ),
                ),
            ],
        ],
        0.49,
        'Fraction of Weyl chamber reached with 2 gates: 0.600\nFraction of Weyl chamber reached with 2 gates and 3 gates(same single qubit): 1.000',
        (),
    )


def get_XEBPhasedFSimCalibrationOptions():
    return cirq.google.XEBPhasedFSimCalibrationOptions(
        n_library_circuits=20,
        n_combinations=10,
        cycle_depths=(5, 25, 50, 100, 200, 300),
        fatol=0.005,
        xatol=0.005,
        fsim_options=cirq.experiments.XEBPhasedFSimCharacterizationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=True,
            characterize_gamma=True,
            characterize_phi=True,
            theta_default=0.0,
            zeta_default=0.0,
            chi_default=0.0,
            gamma_default=0.0,
            phi_default=0.0,
        ),
    )


def get_XEBPhasedFSimCalibrationRequest():
    return cirq.google.XEBPhasedFSimCalibrationRequest(
        pairs=(
            (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            (cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
        ),
        gate=cirq.FSimGate(theta=0.7853981633974483, phi=0.0),
        options=cirq.google.XEBPhasedFSimCalibrationOptions(
            n_library_circuits=20,
            n_combinations=10,
            cycle_depths=(5, 25, 50, 100, 200, 300),
            fatol=0.005,
            xatol=0.005,
            fsim_options=cirq.experiments.XEBPhasedFSimCharacterizationOptions(
                characterize_theta=True,
                characterize_zeta=True,
                characterize_chi=True,
                characterize_gamma=True,
                characterize_phi=True,
                theta_default=0.0,
                zeta_default=0.0,
                chi_default=0.0,
                gamma_default=0.0,
                phi_default=0.0,
            ),
        ),
    )


def get_Bristlecone():
    return cirq_google.Bristlecone


def get_PhasedFSimCalibrationResult():
    return cirq_google.PhasedFSimCalibrationResult(
        parameters={
            (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): cirq_google.PhasedFSimCharacterization(
                theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
            ),
            (cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)): cirq_google.PhasedFSimCharacterization(
                theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
            ),
        },
        gate=cirq.FSimGate(theta=0.7853981633974483, phi=0.0),
        options=cirq_google.FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
            readout_error_tolerance=0.4,
        ),
        project_id='project_id',
        program_id='program_id',
        job_id='job_id',
    )


def get_Calibration():
    return cirq_google.Calibration(
        metrics={
            'xeb': {
                (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.9999],
                (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)): [0.9998],
            },
            't1': {
                (cirq.GridQubit(0, 0),): [321],
                (cirq.GridQubit(0, 1),): [911],
                (cirq.GridQubit(1, 0),): [505],
            },
            'globalMetric': {(): ['abcd']},
        }
    )


def get_CalibrationTag():
    return cirq_google.CalibrationTag('xeb')


def get_PhasedFSimCharacterization():
    return cirq_google.PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5)


def get_LocalXEBPhasedFSimCalibrationRequest():
    return cirq_google.LocalXEBPhasedFSimCalibrationRequest(
        pairs=(
            (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            (cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
        ),
        gate=cirq.FSimGate(theta=0.7853981633974483, phi=0.0),
        options=cirq_google.LocalXEBPhasedFSimCalibrationOptions(
            n_library_circuits=20,
            n_combinations=10,
            cycle_depths=(5, 25, 50, 100, 200, 300),
            fatol=0.005,
            xatol=0.005,
            fsim_options=cirq.experiments.XEBPhasedFSimCharacterizationOptions(
                characterize_theta=True,
                characterize_zeta=True,
                characterize_chi=True,
                characterize_gamma=True,
                characterize_phi=True,
                theta_default=0.0,
                zeta_default=0.0,
                chi_default=0.0,
                gamma_default=0.0,
                phi_default=0.0,
            ),
            n_processes=None,
        ),
    )


def get_PhysicalZTag():
    return cirq_google.PhysicalZTag()


def get_CalibrationLayer():
    return cirq_google.CalibrationLayer(
        calibration_type='xeb',
        program=cirq.Circuit(
            [
                cirq.Moment(
                    cirq.X(cirq.GridQubit(1, 1)),
                ),
            ]
        ),
        args={'type': 'full', 'samples': 100},
    )


def get_SYC():
    return cirq_google.SYC


def get_CalibrationResult():
    return cirq_google.CalibrationResult(
        code=1,
        error_message='a',
        token='b',
        valid_until=datetime.datetime(2020, 11, 25, 15, 59, 44, 41021),
        metrics=cirq_google.Calibration(
            metrics={
                'xeb': {
                    (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.9999],
                    (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)): [0.9998],
                },
                't1': {
                    (cirq.GridQubit(0, 0),): [321],
                    (cirq.GridQubit(0, 1),): [911],
                    (cirq.GridQubit(1, 0),): [505],
                },
                'globalMetric': {(): ['abcd']},
            }
        ),
    )


def get_SycamoreGate():
    return cirq_google.SYC


def get_Foxtail():
    return cirq_google.Foxtail


def get__NamedConstantXmonDevice():
    return cirq_google.Foxtail


def get_list():
    return [cirq_google.Foxtail, cirq_google.Bristlecone]


## END DEFS
GET_FNS = {
    'LocalXEBPhasedFSimCalibrationOptions': get_LocalXEBPhasedFSimCalibrationOptions,
    'FloquetPhasedFSimCalibrationRequest': get_FloquetPhasedFSimCalibrationRequest,
    'CouplerPulse': get_CouplerPulse,
    'FloquetPhasedFSimCalibrationOptions': get_FloquetPhasedFSimCalibrationOptions,
    'GateTabulation': get_GateTabulation,
    'XEBPhasedFSimCalibrationOptions': get_XEBPhasedFSimCalibrationOptions,
    'XEBPhasedFSimCalibrationRequest': get_XEBPhasedFSimCalibrationRequest,
    'Bristlecone': get_Bristlecone,
    'PhasedFSimCalibrationResult': get_PhasedFSimCalibrationResult,
    'Calibration': get_Calibration,
    'CalibrationTag': get_CalibrationTag,
    'PhasedFSimCharacterization': get_PhasedFSimCharacterization,
    'LocalXEBPhasedFSimCalibrationRequest': get_LocalXEBPhasedFSimCalibrationRequest,
    'PhysicalZTag': get_PhysicalZTag,
    'CalibrationLayer': get_CalibrationLayer,
    'SYC': get_SYC,
    'CalibrationResult': get_CalibrationResult,
    'SycamoreGate': get_SycamoreGate,
    'Foxtail': get_Foxtail,
}

## END DICT

PATH = '../cirq-google/cirq_google/json_test_data'


def autogenerate():
    lines = []
    dict_lines = ['GET_FNS = {']
    fns = glob.glob(f'{PATH}/*.repr')

    with open(__file__) as f:
        existing = f.read()

    imports, import_marker, rest = existing.partition('##' + ' END IMPORTS')
    defs, def_marker, rest = rest.partition('##' + ' END DEFS')
    _, dict_marker, rest = rest.partition('##' + ' END DICT')

    for fn in fns:
        oname = os.path.basename(fn)
        oname, _ = os.path.splitext(oname)

        dict_lines += [
            f"    '{oname}': get_{oname},"
        ]

        if f'def get_{oname}():' in existing:
            continue

        with open(fn) as f:
            rep = f.read()

        rep = rep.splitlines()
        rep = rep[:1] + ['    ' + r for r in rep[1:]]
        rep = '\n'.join(rep)
        lines += [
            f'def get_{oname}():',
            f'    return {rep}',
            '', '',
        ]

    dict_lines += ['}', '']

    with open(__file__, 'w') as f:
        f.write(imports + import_marker)
        f.write(defs)
        f.write('\n'.join(lines))
        f.write(def_marker + '\n')
        f.write('\n'.join(dict_lines))
        f.write(dict_marker)
        f.write(rest)


def main():
    for fn in glob.glob(f'{PATH}/*.repr'):
        os.remove(fn)
    for fn in glob.glob(f'{PATH}/*.json'):
        os.remove(fn)

    for oname, getter in GET_FNS.items():
        o = getter()
        with open(f'{PATH}/{oname}.repr', 'w') as f:
            f.write(repr(o) + '\n')
        cirq.to_json(o, f'{PATH}/{oname}.json')


if __name__ == '__main__':
    main()
