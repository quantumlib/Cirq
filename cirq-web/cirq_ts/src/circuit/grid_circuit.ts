// Copyright 2021 The Cirq Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Group} from 'three';
import {GridQubit} from './components/grid_qubit';
import {TwoQubitGate, SingleQubitGate, GridCoord} from './components/types';

/**
 * Class that gathers serialized circuit information
 * from a Python cirq.Circuit and reconstructs it to be 
 * displayed using three.js
 */
export class GridCircuit extends Group {
    readonly moments: number;
    private circuit: WeakMap<[number, number], GridQubit>;

    /**
     * Class constructor
     * @param moments The number of moments of the circuit. This 
     * determines the length of all the qubit lines in the diagram.
     * @param qubits A list of GridCoord objects representing the locations of the 
     * qubits in the circuit. 
     */
    constructor(moments: number, qubits: GridCoord[]) {
        super();
        this.moments = moments;
        this.circuit = new WeakMap();

        for (const coord of qubits) {
            this.addQubit(coord.row, coord.col);
        }
    }

    /**
     * Adds Gate objects to class circuit map and generates three.js
     * objects, adding them to the group.
     * @param list A list of Gate objects to be added to the circuit.
     */
    displayGatesFromList(list: (SingleQubitGate | TwoQubitGate)[]) {
        for (const gate of list) {
            switch (gate.type) {
                case 'SingleQubitGate':
                    this.addSingleQubitGate(gate);
                    break;
                case 'TwoQubitGate':
                    this.addTwoQubitGate(gate);
                    break;
            }
        } 
    }

    /**
     * Adds a single qubit gate to the circuit map, and the corresponding
     * three.js object to the group.
     * @param gate The SingleQubitGate object to be added.
     */
    addSingleQubitGate(gate: SingleQubitGate) {
        //.get gives a reference to an object, so we're good to
        // just modify
        const qubit = this.circuit.get([gate.row, gate.col])!;
        qubit.addSingleQubitGate(gate.text, gate.color, gate.moment);
    }

    /**
     * Adds a two qubit gate to the circuit map, and the corresponding
     * three.js object to the group.
     * @param gate The TwoQubitGate object to be added.
     */
    addTwoQubitGate(gate: TwoQubitGate) {
        const control = this.circuit.get([gate.row, gate.col])!;
        control.addControl(gate.moment);
        control.addLineToQubit(gate.targetGate.row, gate.targetGate.col, gate.moment);

        const target = this.circuit.get([gate.targetGate.row, gate.targetGate.col])!;
        target.addSingleQubitGate(gate.targetGate.text, gate.targetGate.color, gate.targetGate.moment);
    }

    private addQubit(x: number, y: number) {
        const qubit = new GridQubit(x, y, this.moments);
        this.circuit.set([x, y], qubit)
        this.add(qubit);
    }
}
