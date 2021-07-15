export interface Gate {
    readonly moment: number;
}

export interface SingleQubitGate extends Gate {
    readonly text: string;
    readonly color: string;
    readonly row: number;
    readonly col: number;
    readonly type: 'SingleQubitGate';
}

export interface ControlledGate extends Gate {
    readonly row: number;
    readonly col: number;
    readonly targetGate: SingleQubitGate;
    readonly type: 'ControlledGate';
}




