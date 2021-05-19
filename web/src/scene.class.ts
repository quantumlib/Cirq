import {Scene, PerspectiveCamera, WebGLRenderer, Camera, Object3D} from 'three';

export class Cirq3DScene {
  private static readonly VIZ_WIDTH: number = window.innerWidth / 2;
  private static readonly VIZ_HEIGHT: number = window.innerHeight / 2;

  private scene: Scene;
  public camera: Camera;
  public renderer: WebGLRenderer;

  public constructor(
    fov = 75,
    aspect: number = Cirq3DScene.VIZ_HEIGHT / Cirq3DScene.VIZ_WIDTH,
    near = 0.1,
    far = 1000
  ) {
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(fov, aspect, near, far);
    this.renderer = new WebGLRenderer();
    this.renderer.setSize(Cirq3DScene.VIZ_WIDTH, Cirq3DScene.VIZ_HEIGHT);

    this.init();
  }

  private init() {
    this.camera.position.z = 5;
    this.addSceneToHTML();
    this.setRenderSize();
    this.animate();
  }

  private addSceneToHTML() {
    // We must assume that there is already a div element named 'container'
    const container = document.getElementById('container')!;
    container.appendChild(this.renderer.domElement);
  }

  public setRenderSize(
    width: number = Cirq3DScene.VIZ_WIDTH,
    height: number = Cirq3DScene.VIZ_HEIGHT
  ) {
    this.renderer.setSize(width, height);
  }

  public animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.renderer.render(this.scene, this.camera);
  }

  // Overloaded method
  public add(object: Object3D) {
    this.scene.add(object);
  }
}
