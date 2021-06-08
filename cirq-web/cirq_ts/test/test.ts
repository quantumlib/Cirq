import {Vector3} from 'three';
const assert = require('assert');

describe('The THREE object', function() {
    it('should be able to construct a Vector3 with default of x=0', function() {
      const vec3 = new Vector3();
      assert.equal(0, vec3.x);
    })
})