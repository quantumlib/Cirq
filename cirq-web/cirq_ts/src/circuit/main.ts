import {Scene, PerspectiveCamera, WebGLRenderer, Raycaster, Vector2, AxesHelper, Vector3} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {Circuit} from './circuit';
import {SingleQubitGate, ControlledGate} from './components/types';

const mouse = new Vector2();
const raycaster = new Raycaster();

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

function onMouseMove( event: any ) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY/window.innerHeight) * 2 + 1;
}

// // Raycasting
// function addRaycasting() {
//     raycaster.setFromCamera(mouse, camera);
//     const intersects = raycaster.intersectObjects(scene.children);
//     for (let i = 0; i < intersects.length; i++){
//         console.log(intersects[i]);
//         //intersects[i].object
//     }
//     requestAnimationFrame(addRaycasting);
//     renderer.render(scene, camera);
// }


// render();


export function createGridCircuit(qubits: number[][], numMoments: number, sceneId: string): Circuit {
    const scene = createAndRenderScene(qubits.length, sceneId);
    addEventListener('mousemove', onMouseMove, false);

    const circuit = new Circuit(numMoments);
    console.log(qubits);
    for (const qubit of qubits) {
        circuit.addQubit(qubit[0], qubit[1]);
    }
    //console.log(circuit);
    scene.add(circuit);

    return circuit;
}

