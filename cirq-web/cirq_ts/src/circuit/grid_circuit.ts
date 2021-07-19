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

class CircuitMap {
    private map = new Map<string, GridQubit>();

    set(key: [number, number], value: GridQubit): this {
        const keyAsString = key.join(',');
        this.map.set(keyAsString, value);
        return this;
    }

    get(key: [number, number]) : GridQubit | undefined {
        const keyAsString = key.join(',');
        return this.map.get(keyAsString);
    }

    get size() {
        return this.map.size;
    }

    // forEach(
    //     callback: (
    //         value: GridQubit, 
    //         key: [number, number], 
    //         map: Map<[number, number], GridQubit>
    //         ) => void, 
    //     thisArg?: any
    //     ): void {
    //     this.map.forEach((value, key) => {
    //         const keyAsArr = key.split(',').map(Number) as [number, number];
    //         callback.call(thisArg, value, keyAsArr, this);        
    //     })
    // }
}

export class GridCircuit extends Group {
    readonly moments: number;
    readonly circuit: CircuitMap;

    constructor(moments: number, qubits: GridCoord[]) {
        super();
        this.moments = moments;
        this.circuit = new CircuitMap();

        for (const coord of qubits) {
            this.addQubit(coord.row, coord.col);
        }
    }

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

    addSingleQubitGate(gate: SingleQubitGate) {
        //.get gives a reference to an object, so we're good to
        // just modify
        const qubit = this.circuit.get([gate.row, gate.col])!;
        qubit.addSingleQubitGate(gate.text, gate.color, gate.moment);
    }

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
