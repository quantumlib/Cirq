import {assert, expect} from 'chai';
import {loadAndDisplayText} from './text';

describe('Text methods', function() {

    const textItems = loadAndDisplayText();

  it('returns a list of Mesh objects', function() {   
    for (const text of textItems) {
        expect(text.type).to.equal("Mesh")
    }
  })

  it('returns 6 valid labels, one for each state', function() {
    expect(textItems.length).to.equal(6);
  })
})