import {BufferGeometry, Group, LineBasicMaterial, Vector3} from 'three';
import {GridQubit} from './components/grid_qubit';

export class Circuit extends Group {
    public rows: number;
    public cols: number;
    public moments: number;
    public circuit: any;

    constructor(rows: number, cols: number, moments: number) {
        super();

        this.rows = rows;
        this.cols = cols;
        this.moments = moments;
        this.circuit = [];
        for (var i=0; i< rows; i++) {
            this.circuit[i]=[];
            for (var j=0; j<cols; j++){
                this.circuit[i][j]=[];
                for (var k=0; k < cols; k++) {
                    this.circuit[i][j][k] = 0;
                }
            }
        }
    }

    generateQubitGrid(){
        for (let x = 0; x < this.rows; x++) {
            for (let y = 0; y < this.cols; y++) {
                const qubit = new GridQubit(x, y, this.moments);
                this.circuit[x][y] = qubit;
                this.add(qubit);
            }
        }
        return this;
    }

    addCube(x: number, y: number, moment: number) {
        this.circuit[x][y].addBox(moment);
    }

    addCNOT(x1: number, y1: number, x2: number, y2: number, moment: number) {
        this.circuit[x1][y1].addCNOT(x2, y2, moment);
    }
}
