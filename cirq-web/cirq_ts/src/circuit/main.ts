import {Scene, PerspectiveCamera, WebGLRenderer, Raycaster, Vector2, AxesHelper, Vector3} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {Circuit} from './circuit';


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
    for (const qubit of qubits) {
        circuit.addQubit(qubit[0], qubit[1]);
    }
    console.log(scene);
    scene.add(circuit);

    return circuit;
}



// const circuit = new Circuit(NUM_QUBITS, NUM_QUBITS, 5);
// const grid = circuit.generateQubitGrid();
// console.log(grid);
// scene.add(grid);

// Params for addCube (qubitX, qubitY, momentID)
//circuit.addCube(1, 0, 1);

/*
//Params for addCNOT (ctrlX, ctrlY, targetX, targetY, MomentID)
circuit.addCNOT(0, 0, 0, 1, 1)

circuit.addCube(1, 0, 2);
circuit.addCNOT(0, 0, 0, 1, 2)

for(let i = 0; i < 20; i++){
    const randMoment = Math.floor(Math.random()*4);
    const rndIntX = Math.floor(Math.random() * 4);
    const rndIntY = Math.floor(Math.random() * 4);
    circuit.addCube(rndIntX, rndIntY, randMoment);
    circuit.addCNOT(rndIntX, rndIntY+1, rndIntX+1, rndIntY+1, randMoment+1);
}
*/