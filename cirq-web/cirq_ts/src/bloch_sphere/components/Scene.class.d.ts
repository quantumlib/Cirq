import { WebGLRenderer, Camera, Object3D } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
export declare class BlochSphereScene {
    private static readonly VIZ_WIDTH;
    private static readonly VIZ_HEIGHT;
    private scene;
    camera: Camera;
    renderer: WebGLRenderer;
    controls: OrbitControls;
    constructor(fov?: number, aspect?: number, near?: number, far?: number);
    private init;
    private setUpControls;
    private addSceneToHTML;
    setRenderSize(width?: number, height?: number): void;
    animate(): void;
    add(object: Object3D): void;
}
