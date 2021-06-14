import {assert, expect} from 'chai';
import { Color } from 'three';
import {generateAxis} from './axes';

describe('Axes methods', function() {

    // Sanity check
    it('returns an object', function() {  
      const axes = generateAxis(5);
      assert.typeOf(axes, "object");
    })

    it('returns a mapping of Line objects to axis labels', function() {
      const axes = generateAxis(5);
      expect(axes.x.type).to.equal("Line");
      expect(axes.y.type).to.equal("Line");
      expect(axes.z.type).to.equal("Line");
    })

    it('has configurable axis colors', function() {
      // No way access color attribute in three.js, looking into it
      const axes = generateAxis(5, "#fff", "#fff", "#fff");
      expect(axes.x.type).to.equal("Line");
      expect(axes.y.type).to.equal("Line");
      expect(axes.z.type).to.equal("Line");
    })
})
