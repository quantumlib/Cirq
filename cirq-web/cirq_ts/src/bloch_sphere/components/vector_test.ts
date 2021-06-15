import {expect} from 'chai';
import {createVector} from './vector';

describe('Vector methods', () => {
  it('returns an ArrowHelper type', () => {
    const vector = createVector();
    expect(vector.type).to.equal('ArrowHelper');
  });

  it('can parse correct JSON data', () => {
    const jsonString = '{"x": 0, "y": 1, "z": 2}';
    const vector = createVector(jsonString);
    expect(vector.type).to.equal('ArrowHelper');
  });
});
