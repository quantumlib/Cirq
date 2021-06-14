import {assert, expect} from 'chai';
import { ArrowHelper } from 'three';
import {createVector} from './vector';

describe('Vector methods', function() {

  it('returns an ArrowHelper type', function() {   
    const vector = createVector();
    expect(vector.type).to.equal('ArrowHelper');
  })

  it('can parse correct JSON data', function() {
    const jsonString = '{"x": 0, "y": 1, "z": 2}';
    const vector = createVector(jsonString);
    expect(vector.type).to.equal('ArrowHelper');
  })
})