if (typeof require === 'function') // test for nodejs environment
{
  var THREE = require('three');
}

export class CirqSphere {

    static CreateSphere() {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);

        const container = document.getElementById("container")!;
        container.appendChild(renderer.domElement);

        const geometry = new THREE.SphereGeometry();
        const material = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
        const sphere = new THREE.Mesh( geometry, material );
        scene.add( sphere );

        camera.position.z = 5;
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }

        animate();
    }

}
