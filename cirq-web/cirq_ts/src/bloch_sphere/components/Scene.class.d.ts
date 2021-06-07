import { WebGLRenderer, Camera, Object3D } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
export declare class BlochSphereScene {
    private static readonly VIZ_WIDTH;
    private static readonly VIZ_HEIGHT;
    private scene;
    camera: Camera;
    renderer: WebGLRenderer;
    controls: OrbitControls;
    /**
     * Initializes a 3D Scene proportional to the bloch sphere visualzation.
     * @param fov The vertical field of view for the Scene's Perspective Camera
     * @param aspect The aspect ratio for the Scene's Perspective Camera
     * @param near The near plane for the Scene's Perspective Camera
     * @param far The far plane for the Scene's Perspective Camera
     */
    constructor(fov?: number, aspect?: number, near?: number, far?: number);
    /**
     * Initialization helper function. Also sets the starting
     * position of the camera.
     */
    private init;
    /**
     * Handles setting up the controls for OrbitControls.
     */
    private setUpControls;
    /**
     * Adds the 3D Scene to the HTML. Currently locked onto
     * the 'container' div.
     */
    private addSceneToHTML;
    /**
     * Configures the output canvas and the viewport
     * to make sure that elements are rendered correctly
     * onto the scene.
     * @param width The desired width of the rendering canvas.
     * @param height The desired height of the rendering canvas.
     */
    setRenderSize(width?: number, height?: number): void;
    /**
     * Enables interactivity for the visualization.
     */
    animate(): void;
    /**
     * Overloaded method. Adds an object to the
     * scene.
     * @param object The object to be added to the scene.
     */
    add(object: Object3D): void;
}
