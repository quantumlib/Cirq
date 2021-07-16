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

import {
    Vector3,
    Line,
    LineBasicMaterial,
    BufferGeometry,
    Texture,
    SpriteMaterial,
    Sprite,
    Mesh,
    MeshBasicMaterial,
    SphereGeometry,
    CylinderGeometry,
    DoubleSide,
    BoxGeometry,
} from 'three';

export class ConnectionLine extends Line {
    constructor(startCoord: Vector3, endCoord: Vector3) {
        super();
        const COLOR : string = 'black';

        const material = new LineBasicMaterial({color: COLOR});
        const points = [startCoord, endCoord];
        const geometry = new BufferGeometry().setFromPoints(points);
        return new Line(geometry, material);
    }
}


export class QubitLabel extends Sprite {
    constructor(text: string){
        super();

        const CANVAS_SIZE = 128;

        const canvas = document.createElement('canvas');
        canvas.width = CANVAS_SIZE;
        canvas.height = CANVAS_SIZE;
        // Allows us to keep track of what we're adding to the
        // canvas.
        canvas.textContent = text;
      
        const context = canvas.getContext('2d')!;
        context.fillStyle = '#000000';
        context.textAlign = 'center';
        context.font = '20px Arial';
        context.fillText(text, CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      
        const map = new Texture(canvas);
        map.needsUpdate = true;
      
        const material =  new SpriteMaterial({
          map: map,
          transparent: true, // for a transparent canvas background
        });

        const sprite = new Sprite(material);
        return sprite;
    }
}

export class QubitLine extends Line {
    constructor(startCoord: Vector3, endCoord: Vector3) {
        super();
        const COLOR : string = 'gray';

        const material = new LineBasicMaterial({color: COLOR});
        const points = [startCoord, endCoord];
        const geometry = new BufferGeometry().setFromPoints(points);
        return new Line(geometry, material);
    }
}

export class Control3DSymbol extends Mesh {
    constructor() {
        super();
        const COLOR: string = 'black';

        const material = new MeshBasicMaterial({color: COLOR});
        const geometry = new SphereGeometry(0.1, 32, 32);
        const sphere = new Mesh(geometry, material);

        return sphere;
    }
}

export class X3DSymbol extends Mesh {
    constructor(color: string) {
        super();

        const material = new MeshBasicMaterial({color: color, side: DoubleSide});
        const geometry = new CylinderGeometry(0.3, 0.3, 0.1, 32,1, true, 0, 2*Math.PI );
        const cylinder = new Mesh(geometry, material);
        return cylinder;
    }
}

export class BoxGate3DSymbol extends Mesh {
    constructor(label: string, color: string) {
        super();

        var canvas = document.createElement("canvas")!;
        var context = canvas.getContext("2d")!;
        canvas.width = canvas.height = 128;

        context.fillStyle = color;
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = "50pt arial bold";
        context.fillStyle = "black";
        context.fillText(label, canvas.width/2 - 25, canvas.height/2 + 25);

        const map = new Texture(canvas);
        map.needsUpdate = true;

        const geometry = new BoxGeometry(0.5, 0.5, 0.5);
        const material = new MeshBasicMaterial( {map: map, color: color } );
        const cube = new Mesh( geometry, material );

        return cube;
    }
}
