import {Scene, PerspectiveCamera, WebGLRenderer, Raycaster, Vector2, AxesHelper, Vector3} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {GridCircuit} from './grid_circuit';
import {GridCoord} from './components/types';

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

export function createGridCircuit(qubits: GridCoord[], numMoments: number, sceneId: string): GridCircuit {
    const scene = createAndRenderScene(qubits.length, sceneId);

    const circuit = new GridCircuit(numMoments, qubits);
    scene.add(circuit);

    return circuit;
}

