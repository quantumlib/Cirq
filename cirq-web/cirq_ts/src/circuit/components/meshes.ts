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
  Group,
  Color,
} from 'three';

/**
 * A wrapper for a three.js Line object representing a connection line
 * between two qubits. Useful when building controlled gates.
 */
export class ConnectionLine extends Line {
  /**
   * Class constructor.
   * @param startCoord The starting coordinate of the line
   * @param endCoord The ending coordinate of the line
   * @returns a CollectionLine object that can be added to a three.js scene
   */
  constructor(startCoord: Vector3, endCoord: Vector3) {
    const COLOR = 'black';

    const material = new LineBasicMaterial({color: COLOR});
    const points = [startCoord, endCoord];
    const geometry = new BufferGeometry().setFromPoints(points);

    super(geometry, material);
    return this;
  }
}

/**
 * A wrapper for a three.js Sprite object which is used to label the
 * location of specific qubits.
 */
export class QubitLabel extends Sprite {
  readonly text: string;
  /**
   * Class constructor.
   * @param text The text which the label should display
   * @returns a QubitLabel object that can be added to a three.js scene
   */
  constructor(text: string) {
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

    const material = new SpriteMaterial({
      map,
      transparent: true, // for a transparent canvas background
    });

    super(material);
    this.text = text;
    return this;
  }
}

/**
 * A wrapper for a three.js Line object which represents a qubit in
 * this circuit.
 */
export class QubitLine extends Line {
  /**
   * Class constructor.
   * @param startCoord The starting coordinate of the line
   * @param endCoord The ending coordinate of the line
   * @returns a QubitLine object that can be added to a three.js scene.
   */
  constructor(startCoord: Vector3, endCoord: Vector3) {
    const COLOR = 'gray';

    const material = new LineBasicMaterial({color: COLOR});
    const points = [startCoord, endCoord];
    const geometry = new BufferGeometry().setFromPoints(points);

    super(geometry, material);
    return this;
  }
}

/**
 * A wrapper for a three.js Sphere which represents a control
 * in a circuit.
 */
export class Control3DSymbol extends Mesh {
  /**
   * Class constructor.
   * @returns a Control3DSymbol object that can be added to a three.js scene.
   */
  constructor() {
    const COLOR = 'black';

    const material = new MeshBasicMaterial({color: COLOR});
    const geometry = new SphereGeometry(0.1, 32, 32);

    super(geometry, material);
    return this;
  }
}

/**
 * A wrapper for a three.js Group which represents an X operation
 * in a circuit.
 */
export class X3DSymbol extends Group {
  /**
   * Class constructor.
   * @param color The color of the symbol
   * @returns an X3DSymbol object that can be added to a three.js scene
   */
  constructor(color: string) {
    super();
    const material = new MeshBasicMaterial({color: color, side: DoubleSide});
    const geometry = new CylinderGeometry(
      0.3,
      0.3,
      0.1,
      32,
      1,
      true,
      0,
      2 * Math.PI
    );
    const hollowCylinder = new Mesh(geometry, material);
    this.add(hollowCylinder);

    // Creates the "X" in the middle of the holow cylinder
    const rotationAngle = Math.PI / 2;

    const xLineMaterial = new MeshBasicMaterial({color: color});
    const xLineGeometry = new CylinderGeometry(0.01, 0.01, 0.6);
    const xLine = new Mesh(xLineGeometry, xLineMaterial);
    xLine.rotation.x = rotationAngle;

    const zLineMaterial = new MeshBasicMaterial({color: color});
    const zLineGeometry = new CylinderGeometry(0.01, 0.01, 0.6);
    const zLine = new Mesh(zLineGeometry, zLineMaterial);
    zLine.rotation.z = rotationAngle;

    this.add(xLine);
    this.add(zLine);
    return this;
  }
}

/**
 * A wrapper for a three.js Group which represents an Swap operation
 * in a circuit.
 */
export class Swap3DSymbol extends Group {
  /**
   * Class constructor.
   * @returns a Swap3DSymbol object that can be added to a three.js scene
   */
  constructor() {
    super();

    const xLineMaterial = new MeshBasicMaterial({color: 'black'});
    const xLineGeometry = new CylinderGeometry(0.01, 0.01, 0.3);
    const xLine = new Mesh(xLineGeometry, xLineMaterial);
    xLine.rotation.x = Math.PI / 2;
    xLine.rotation.z = (3 * Math.PI) / 4;

    const zLineMaterial = new MeshBasicMaterial({color: 'black'});
    const zLineGeometry = new CylinderGeometry(0.01, 0.01, 0.3);
    const zLine = new Mesh(zLineGeometry, zLineMaterial);
    zLine.rotation.x = Math.PI / 2;
    zLine.rotation.z = Math.PI / 4;

    this.add(xLine);
    this.add(zLine);
    return this;
  }
}

/**
 * A wrapper for a three.js Box which represents arbitrary single qubit
 * operations in a circuit
 */
export class BoxGate3DSymbol extends Mesh {
  /**
   * Class constructor.
   * @param label The label that will go on the three.js box
   * @param color The color of the box
   * @returns a BoxGate3DSymbol object that can be added to a three.js scene
   */
  constructor(label: string, color: string) {
    const canvas = document.createElement('canvas')!;
    const context = canvas.getContext('2d')!;
    canvas.width = canvas.height = 128;

    context.fillStyle = color;
    context.fillRect(0, 0, canvas.width, canvas.height);

    let fontSize = 60;
    let textWidth;
    do {
      fontSize /= 1.2;
      context.font = `${fontSize}pt arial bold`;
      textWidth = context.measureText(label).width;
    } while (textWidth > canvas.width);

    const hsl = new Color(color).getHSL({h: 0, s: 0, l: 0});
    context.fillStyle = hsl.l < 0.5 ? 'white' : 'black';
    context.fillText(
      label,
      canvas.width / 2 - textWidth / 2,
      canvas.height / 2 + fontSize / 2
    );

    const map = new Texture(canvas);
    map.needsUpdate = true;

    const geometry = new BoxGeometry(0.5, 0.5, 0.5);
    const material = new MeshBasicMaterial({map: map, color: 'white'});

    super(geometry, material);
    return this;
  }
}
