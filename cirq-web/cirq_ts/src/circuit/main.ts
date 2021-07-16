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

import {Scene, PerspectiveCamera, WebGLRenderer} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {GridCircuit} from './grid_circuit';
import {GridCoord} from './components/types';

function createAndRenderScene(numQubits: number, sceneId: string): any {
    const WIDTH = 1000;
    const HEIGHT = 700;
    const NUM_QUBITS = numQubits;

    const scene = new Scene();
    const camera = new PerspectiveCamera( 75, WIDTH / HEIGHT, 0.1, 1000 );

    const renderer = new WebGLRenderer({alpha: true});
    const controls = new OrbitControls( camera, renderer.domElement );

    renderer.setSize( WIDTH, HEIGHT );
    const el = document.getElementById(sceneId)!;

    el.appendChild( renderer.domElement );

    camera.position.x = NUM_QUBITS / 2 - 1;
    camera.position.z = NUM_QUBITS / 2 - 1;
    camera.position.y = 2.5;
    
    controls.target.set(NUM_QUBITS / 2 - 1, 2.5, NUM_QUBITS / 2 - 1);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 5;
    controls.maxDistance = 200;
    controls.maxPolarAngle = Math.PI;
    
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render( scene, camera );
    }
    animate();

    return scene;
}

export function createGridCircuit(qubits: GridCoord[], numMoments: number, sceneId: string): GridCircuit {
    const scene = createAndRenderScene(qubits.length, sceneId);

    const circuit = new GridCircuit(numMoments, qubits);
    scene.add(circuit);

    return circuit;
}

