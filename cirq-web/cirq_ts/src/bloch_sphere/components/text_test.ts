import {expect} from 'chai';
import {loadAndDisplayText} from './text';
import {JSDOM} from 'jsdom';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const { window } = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('Text methods', function() {

  const textItems = loadAndDisplayText();

  it('returns a list of Sprite objects', function() {   
    for (const text of textItems) {
        expect(text.type).to.equal("Sprite")
    }
  })

  it('returns 6 valid labels, one for each state', function() {
    expect(textItems.length).to.equal(6);
  })
})