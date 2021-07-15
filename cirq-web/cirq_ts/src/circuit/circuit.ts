import {Group} from 'three';
import {ControlledGate, SingleQubitGate } from './components/types';
import {GridQubit} from './components/grid_qubit';

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
}

export class Circuit extends Group {
    public moments: number;
    public circuit: any;

    constructor(moments: number) {
        super();
        this.moments = moments;
        this.circuit = new CircuitMap();
    }

    addQubit(x: number, y: number) {
        const qubit = new GridQubit(x, y, this.moments);
        this.circuit.set([x, y], qubit)
        this.add(qubit);
    }

    displayGatesFromList(list: (SingleQubitGate | ControlledGate)[]) {
        for (const gate of list) {
            switch (gate.type) {
                case 'SingleQubitGate':
                    this.addSingleQubitGate(gate);
                    break;
                case 'ControlledGate':
                    this.addControlledGate(gate);
                    break;
            }
        } 
    }

    addSingleQubitGate(gate: SingleQubitGate) {
        //.get gives a reference to an object, so we're good to
        // just modify
        const qubit = this.circuit.get([gate.row, gate.col]);
        qubit.addSingleQubitGate(gate.text, gate.color, gate.moment);
    }

    addControlledGate(gate: ControlledGate) {
        const control = this.circuit.get([gate.row, gate.col]);
        control.addControl(gate.moment);
        control.addLineToQubit(gate.targetGate.row, gate.targetGate.col, gate.moment);

        const target = this.circuit.get([gate.targetGate.row, gate.targetGate.col]);
        target.addSingleQubitGate(gate.targetGate.text, gate.targetGate.color, gate.targetGate.moment);
    }
}
