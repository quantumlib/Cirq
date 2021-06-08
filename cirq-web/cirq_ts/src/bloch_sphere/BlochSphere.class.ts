import {Sphere} from './components/Sphere.class';
import {Axes} from './components/Axes.class';
import {Meridians} from './components/Meridians.class';
import {Text} from './components/Text.class';

import {Group} from 'three';

/**
 * Class bringinging together the individual components like the
 * Sphere, Axes, Meridicans, and Text into the overall visualization
 * of the Bloch sphere.
 */
export class BlochSphere {
  private radius: number;
  private _group: Group;

  constructor(radius: number) {
    this.radius = radius;
    this._group = new Group();
    this._add3dSphere();
    this._addHorizontalMeridians();
    this._addVerticalMeridians();
    this._addAxes();
    this._addAxes();
    this._loadAndDisplayText();
  }

  /**
   * Returns the the group of three.js components that 
   * make up the Bloch sphere. 
   * @returns A Group object of all the added components of the
   * sphere.
   */
  public createBlochSphere() : Group {
    return this._group;
  }

  private _add3dSphere() {
    const sphere = Sphere.createSphere(this.radius);
    this._group.add(sphere);
  }

  private _addAxes() {
    const axes = Axes.createAxes(this.radius);
    this._group.add(axes.x);
    this._group.add(axes.y);
    this._group.add(axes.z);
  }

  private _addHorizontalMeridians() {
    const meridians = Meridians.createHorizontalChordMeridians(this.radius);
    for (const meridian of meridians) {
      this._group.add(meridian);
    }
  }

  private _addVerticalMeridians() {
    const meridians = Meridians.createVerticalMeridians(this.radius);
    for (const meridian of meridians) {
      this._group.add(meridian);
    }
  }

  private _loadAndDisplayText() {
    const textLabels = Text.loadAndDisplayText();
    for (const label of textLabels) {
      this._group.add(label);
    }
  }
}
