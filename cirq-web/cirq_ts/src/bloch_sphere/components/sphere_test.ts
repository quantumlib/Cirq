import {expect} from 'chai';
import {createSphere} from './sphere';

describe('Sphere methods', () => {
  const DEFAULT_RADIUS = 5;
  const sphere = createSphere(DEFAULT_RADIUS);

  it('returns a Mesh', () => {
    expect(sphere.type).to.equal('Mesh');
  });

  it('returns a transparent sphere', () => {
    expect(sphere.material.transparent).to.equal(true);
  });
});
