import {Scene, PerspectiveCamera, WebGLRenderer, Camera, Object3D} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';

export class BlochSphereScene {
  private static readonly VIZ_WIDTH: number = 500;
  private static readonly VIZ_HEIGHT: number = 500;

  private scene: Scene;
  public camera: Camera;
  public renderer: WebGLRenderer;
  public controls: OrbitControls;

  public constructor(
    fov = 75,
    aspect: number = BlochSphereScene.VIZ_HEIGHT / BlochSphereScene.VIZ_WIDTH,
    near = 0.1,
    far = 1000
  ) {
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(fov, aspect, near, far);
    this.renderer = new WebGLRenderer();
    this.renderer.setSize(
      BlochSphereScene.VIZ_WIDTH,
      BlochSphereScene.VIZ_HEIGHT
    );
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);

    this.init();
  }

  private init() {
    this.camera.position.x = 0;
    this.camera.position.y = 0;
    this.camera.position.z = 5;

    this.setUpControls();
    this.addSceneToHTML();
    this.setRenderSize();
    this.animate();
  }

  private setUpControls() {
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 50;
    this.controls.maxPolarAngle = Math.PI;
  }

  private addSceneToHTML() {
    // We must assume that there is already a div element named 'container'
    const container = document.getElementById('container')!;
    container.appendChild(this.renderer.domElement);
  }

  public setRenderSize(
    width: number = BlochSphereScene.VIZ_WIDTH,
    height: number = BlochSphereScene.VIZ_HEIGHT
  ) {
    this.renderer.setSize(width, height);
  }

  public animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // Overloaded method
  public add(object: Object3D) {
    this.scene.add(object);
  }
}
