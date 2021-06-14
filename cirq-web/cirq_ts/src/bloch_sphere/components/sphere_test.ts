import {assert, expect} from 'chai';
import {createSphere} from './sphere';

describe('Sphere methods', function() {

  const DEFAULT_RADIUS = 5;
  const sphere = createSphere(DEFAULT_RADIUS);

  it('returns a Mesh', function() {   
    expect(sphere.type).to.equal("Mesh")
  })

  it ('returns a transparent sphere', function() {
    expect(sphere.material.transparent).to.equal(true);
  })
})