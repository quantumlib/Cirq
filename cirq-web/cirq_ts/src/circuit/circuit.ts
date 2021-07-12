import {BufferGeometry, Group, LineBasicMaterial, Vector3} from 'three';
import {ControlledGate, SingleQubitGate } from './components/gates';
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

    displayGatesFromList(list: any[]) {
        const gateList = this.convertGatesString(list);
        for (const gate of gateList) {
            switch (gate.type) {
                case 'SingleQubitGate':
                    this.addSingleQubitGate(gate as SingleQubitGate);
                    break;
                case 'ControlledGate':
                    this.addControlledGate(gate as ControlledGate);
                    break;
            }
        } 
    }

    addSingleQubitGate(gate: SingleQubitGate) {
        //.get gives a reference to an object, so we're good to
        // just modify
        const qubit = this.circuit.get([gate.row, gate.col]);
        qubit.addSingleQubitGate(gate.asString, gate.color, gate.moment);
    }

    addControlledGate(gate: ControlledGate) {
        const control = this.circuit.get([gate.ctrlRow, gate.ctrlCol]);
        control.addControl(gate.moment);
        control.addLineToQubit(gate.targetGate.row, gate.targetGate.col, gate.moment);

        const target = this.circuit.get([gate.targetGate.row, gate.targetGate.col]);
        target.addSingleQubitGate(gate.targetGate.asString, gate.targetGate.color, gate.targetGate.moment);
    }

    private convertGatesString(list: any[]) : any[] {
        let convertedGates : any[] = [];
        for (const item of list) {
            const type = item.shift();
            switch (type) {
                case 'SingleQubitGate':
                    convertedGates.push(this.castSingleQubitGate(...item));
                    break;
                case 'ControlledGate':
                    convertedGates.push(this.castControlledGate(...item));
                    break;
            }
        }
        return convertedGates;
    }
    
    private castSingleQubitGate(asString?: string, color?: string, row?: number, col?: number, moment?: number) : SingleQubitGate {
        return {
            asString,
            color,
            row,
            col,
            moment,
            type: 'SingleQubitGate',
        } as SingleQubitGate;
    }

    private castControlledGate(
        ctrlRow?: number, 
        ctrlCol?: number,
        targetGate?: SingleQubitGate,
        moment?: number,
        ) : ControlledGate {
        return {
            ctrlRow,
            ctrlCol,
            targetGate,
            moment,
            type: 'ControlledGate',
        } as ControlledGate;
    }
}
