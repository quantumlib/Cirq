import {Scene, PerspectiveCamera, WebGLRenderer, Camera, Object3D, Mesh, Color} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';

export class BlochSphereScene {
  private static readonly VIZ_WIDTH: number = 500;
  private static readonly VIZ_HEIGHT: number = 500;

  private scene: Scene;

  /**
   * The following declarations represent the Camera, Renderer,
   * and Controls objects for the scene.
   * camera - the Three.js camera object responsible for seeing the
   *  visualization from different locations
   * renderer - the Three.js renderer object resposible for rendering
   *  the visualization into the browser. We will be using the WebGL
   * renderer
   * controls - the Three.js OrbitControls object reponsible for managing
   *  mouse input and moving the camera, zooming in and out, etc.
   * divId - the id of the HTML div that will contain the scene output
   */
  camera: Camera;
  renderer: WebGLRenderer;
  controls: OrbitControls;
  divId: string;
  textItems: Mesh[];

  /**
   * Initializes a 3D Scene proportional to the bloch sphere visualzation.
   * @param divId The id of the HTML div that will contain the scene output
   * @param fov The vertical field of view for the Scene's Perspective Camera
   * @param aspect The aspect ratio for the Scene's Perspective Camera
   * @param near The near plane for the Scene's Perspective Camera
   * @param far The far plane for the Scene's Perspective Camera
   */
  public constructor(
    divId = 'container',
    fov = 75,
    aspect: number = BlochSphereScene.VIZ_HEIGHT / BlochSphereScene.VIZ_WIDTH,
    near = 0.1,
    far = 1000
  ) {
    this.divId = divId;
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(fov, aspect, near, far);
    this.renderer = new WebGLRenderer({alpha: true});
    this.renderer.setSize(
      BlochSphereScene.VIZ_WIDTH,
      BlochSphereScene.VIZ_HEIGHT
    );
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.textItems = [];

    this.init();
  }

  /**
   * Initialization helper function. Also sets the starting
   * position of the camera.
   */
  private init() {
    this.camera.position.x = 6;
    this.camera.position.y = 2;
    this.camera.position.z = 2;

    this.setUpControls();
    this.addSceneToHTML();
    this.setRenderSize();
    this.animate();
  }

  /**
   * Handles setting up the controls for OrbitControls.
   */
  private setUpControls() {
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 50;
    this.controls.maxPolarAngle = Math.PI;
  }

  /**
   * Adds the 3D Scene to the HTML according to the div id
   * provided in the constructor.
   */
  private addSceneToHTML() {
    const container = document.getElementById(this.divId)!;
    container.appendChild(this.renderer.domElement);
  }

  /**
   * Configures the output canvas and the viewport
   * to make sure that elements are rendered correctly
   * onto the scene.
   * @param width The desired width of the rendering canvas.
   * @param height The desired height of the rendering canvas.
   */
  public setRenderSize(
    width: number = BlochSphereScene.VIZ_WIDTH,
    height: number = BlochSphereScene.VIZ_HEIGHT
  ) {
    this.renderer.setSize(width, height);
  }

  /**
   * Enables interactivity for the visualization.
   */
  public animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Overloaded method. Adds an object to the
   * scene.
   * @param object The object to be added to the scene.
   */
  public add(object: Object3D) {
    this.scene.add(object);
  }
}
