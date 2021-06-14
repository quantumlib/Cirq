import {assert, expect} from 'chai';
import {BlochSphere} from './bloch_sphere';

describe('The BlochSphere class', function() {
  // Sanity check
  it('initializes correctly', function() {
    const bloch_sphere = new BlochSphere(5);
    assert.typeOf(bloch_sphere, "object");
  })

  it('has a default radius of 5', function() {
    const bloch_sphere = new BlochSphere(5);
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(5);
  })

  it('has a configurable radius', function() {
    const bloch_sphere = new BlochSphere(3);
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(3);
  })

  it('returns a Group object on getBlochSphere()', function() {
    const bloch_sphere = new BlochSphere(5);
    const sphere = bloch_sphere.getBlochSphere();
    expect(sphere.type).to.equal("Group");
  })
})
