import {assert, expect} from 'chai';
import {createHorizontalChordMeridians, createVerticalMeridians, createHorizontalCircleMeridians} from './meridians';

describe('Meridians methods', function() {

  const DEFAULT_RADIUS = 5;

  it('createHorizontalChordMeridians() returns an array of Line objects', function() {   
    const meridians = createHorizontalChordMeridians(DEFAULT_RADIUS);
    for (const meridian of meridians){
      expect(meridian.type).to.equal('Line');
    }
  })

  it('createHorizontalCircleMeridians() returns an array of Line objects', function() {
    const meridians = createHorizontalCircleMeridians(DEFAULT_RADIUS);
    for (const meridian of meridians){
      expect(meridian.type).to.equal('Line');
    }
  })

  it('createVerticalMeridians() returns an array of Line objects', function() {
    const meridians = createVerticalMeridians(DEFAULT_RADIUS);
    for (const meridian of meridians){
      expect(meridian.type).to.equal('Line');
    }
  })
})
