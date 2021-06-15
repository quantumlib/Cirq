import {expect} from 'chai';
import {BlochSphere} from './bloch_sphere';

describe('The BlochSphere class', () => {
  // Sanity check
  it('has a default radius of 5', () => {
    const bloch_sphere = new BlochSphere();
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(5);
  });

  it('has a configurable radius', () => {
    const bloch_sphere = new BlochSphere(3);
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(3);
  });

  it('returns a Group object on getBlochSphere()', () => {
    const bloch_sphere = new BlochSphere(5);
    const sphere = bloch_sphere.getBlochSphere();
    expect(sphere.type).to.equal('Group');
  });
});
