import {Vector3} from 'three';
import {expect} from 'chai';

describe('The THREE object', () => {
    it('should be able to construct a Vector3 with default of x=0', function() {
      const vec3 = new Vector3();
      expect(vec3.x).to.equal(0);
    })
})