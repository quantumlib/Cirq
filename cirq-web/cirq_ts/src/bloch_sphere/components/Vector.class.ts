import {ArrowHelper, CylinderGeometry, MeshBasicMaterial, Mesh, Vector3, BufferGeometry, Line, LineBasicMaterial, Matrix4} from 'three';

export class Vector {
    /**
     * Adds a state vector to the bloch sphere.
     * @param vectorData information representing the location of the vector
     * @returns an ArrowHelper object to be rendered by the scene.
     */
    public static createVector(vectorData?: string) : ArrowHelper {
        let inputData;
        if (vectorData) {
            inputData = JSON.parse(vectorData);
        } else {
            inputData = {
                'x': 0,
                'y': 0,
                'z': 0,
                'v_length': 5,
            }
        }

        console.log(`vectorData: ${vectorData}`);
      
        // Create and normalize the new vector
        const dir = new Vector3(inputData.x, inputData.y, inputData.z);
        dir.normalize();
      
        // Set base properties of the vector
        const origin = new Vector3(0, 0, 0);
        const length = inputData.v_length;
        const hex = '#800080';
        const headWidth = 1;
      
        // Create the arrow representation of the vector and add it to the scene
        const arrowHelper = new ArrowHelper(
          dir,
          origin,
          length,
          hex,
          undefined,
          headWidth
        );

        return arrowHelper;
    }
}