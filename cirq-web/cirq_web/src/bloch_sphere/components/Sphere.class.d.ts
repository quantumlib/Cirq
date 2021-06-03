import {Mesh} from 'three';
export declare class Sphere {
  sphere: Mesh;
  constructor(radius: number);
  returnSphere(): Mesh<
    import('three').BufferGeometry,
    import('three').Material | import('three').Material[]
  >;
}
