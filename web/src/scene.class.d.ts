import {WebGLRenderer, Camera, Object3D} from 'three';
export declare class Cirq3DScene {
  private static readonly VIZ_WIDTH;
  private static readonly VIZ_HEIGHT;
  private scene;
  camera: Camera;
  renderer: WebGLRenderer;
  constructor(fov?: number, aspect?: number, near?: number, far?: number);
  private init;
  private addSceneToHTML;
  setRenderSize(width?: number, height?: number): void;
  animate(): void;
  add(object: Object3D): void;
}
