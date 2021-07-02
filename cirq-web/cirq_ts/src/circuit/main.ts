import {Scene, PerspectiveCamera, WebGLRenderer, AxesHelper, Vector3} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {Circuit} from './circuit';

const scene = new Scene();
const camera = new PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new WebGLRenderer({alpha: true});
const controls = new OrbitControls( camera, renderer.domElement );

renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );


camera.position.x = 2;
camera.position.z = 2;
camera.position.y = 2.5;

controls.target.set(2, 2.5, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 5;
controls.maxDistance = 200;
controls.maxPolarAngle = Math.PI;

function animate() {
	requestAnimationFrame( animate );
    controls.update();
	renderer.render( scene, camera );
}
animate();

const axesHelper = new AxesHelper(3);
//scene.add(axesHelper);
const circuit = new Circuit(5, 5, 5);
const grid = circuit.generateQubitGrid();
console.log(grid);
scene.add(grid);

// Params for addCube (qubitX, qubitY, momentID)
circuit.addCube(1, 0, 1);

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