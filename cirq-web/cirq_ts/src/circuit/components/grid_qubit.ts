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

import {Group, Vector3} from 'three';
import {QubitLine, QubitLabel, X3DSymbol, BoxGate3DSymbol, Control3DSymbol, ConnectionLine} from './meshes';

export class GridQubit extends Group {
    readonly row: number;
    readonly col: number;
    readonly moments: number;

    constructor(row: number, col: number, moments: number) {
        super();

        this.row = row;
        this.col = col;
        this.moments = moments;
        this.createLine();
        this.addLocationLabel();
    }

    private createLine() {
        const coords = [new Vector3(this.row, 0, this.col), new Vector3(this.row, this.moments, this.col)];
        this.add(new QubitLine(coords[0], coords[1]))
    }
    
    private addLocationLabel(){
        const sprite = new QubitLabel(`(${this.row}, ${this.col})`);
        sprite.position.copy(new Vector3(this.row, -0.6, this.col));
        this.add(sprite);
    }

    public addSingleQubitGate(label: string, color: string, moment: number) {
        if (label === 'X') {
            const mesh = new X3DSymbol(color);
            mesh.position.set(this.row, moment, this.col);
            this.add(mesh);
            return;
        }

        const mesh = new BoxGate3DSymbol(label, color);
        mesh.position.set(this.row, moment, this.col);
        this.add(mesh);
    }

    public addControl(moment: number) {
        const mesh = new Control3DSymbol();
        mesh.position.set(this.row, moment, this.col);
        this.add(mesh);

    }

    public addLineToQubit(row: number, col: number, moment: number) {
        const coords = [new Vector3(this.row, moment, this.col), new Vector3(row, moment, col)];
        this.add(new ConnectionLine(coords[0], coords[1]));
    }
}