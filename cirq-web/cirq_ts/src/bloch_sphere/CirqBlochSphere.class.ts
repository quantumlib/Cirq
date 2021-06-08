import {Sphere} from './components/Sphere.class';
import {Axes} from './components/Axes.class';
import {Meridians} from './components/Meridians.class';
import {Text} from './components/Text.class';

import {Group} from 'three';

export class CirqBlochSphere {
  RADIUS: number;
  private _group: Group;

  constructor(radius: number) {
    this.RADIUS = radius;
    this._group = new Group();
    this._add3dSphere();
    this._addHorizontalMeridians();
    this._addVerticalMeridians();
    this._addAxes();
    this._addAxes();
    this._loadAndDisplayText();
  }

  public createSphere() {
    return this._group;
  }

  private _add3dSphere() {
    const sphere = Sphere.createSphere(this.RADIUS);
    this._group.add(sphere);
  }

  private _addAxes() {
    const axes = Axes.createAxes(this.RADIUS);
    this._group.add(axes.x);
    this._group.add(axes.y);
    this._group.add(axes.z);
  }

  private _addHorizontalMeridians() {
    const meridians = Meridians.createHorizontalChordMeridians(this.RADIUS);
    for (const meridian of meridians) {
      this._group.add(meridian);
    }
  }

  private _addVerticalMeridians() {
    const meridians = Meridians.createVerticalMeridians(this.RADIUS);
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
