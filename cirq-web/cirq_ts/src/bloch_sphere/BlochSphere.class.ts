import {Sphere} from './components/Sphere.class';
import {generateAxis} from './components/Axes.class';
import {createHorizontalChordMeridians, createVerticalMeridians} from './components/Meridians.class';
import {Text} from './components/Text.class';

import {Group} from 'three';
import { createVerticalMeridians } from './components/Meridians.class';

/**
 * Class bringinging together the individual components like the
 * Sphere, Axes, Meridicans, and Text into the overall visualization
 * of the Bloch sphere.
 */
export class BlochSphere {
  private radius: number;
  private group: Group;

  constructor(radius: number) {
    this.radius = radius;
    this.group = new Group();
    this.add3dSphere();
    this.addHorizontalMeridians();
    this.addVerticalMeridians();
    this.addAxes();
    this.loadAndDisplayText();
  }

  /**
   * Returns the the group of three.js components that 
   * make up the Bloch sphere. 
   * @returns A Group object of all the added components of the
   * sphere.
   */
  getBlochSphere() : Group {
    // Return a clone of the group to avoid mutation.
    return this.group.clone();
  }

  private add3dSphere() {
    const sphere = Sphere.createSphere(this.radius);
    this.group.add(sphere);
  }

  private addAxes() {
    const axes = generateAxis(this.radius);
    this.group.add(axes.x);
    this.group.add(axes.y);
    this.group.add(axes.z);
  }

  private addHorizontalMeridians() {
    const meridians = createHorizontalChordMeridians(this.radius);
    for (const meridian of meridians) {
      this.group.add(meridian);
    }
  }

  private addVerticalMeridians() {
    const meridians = createVerticalMeridians(this.radius);
    for (const meridian of meridians) {
      this.group.add(meridian);
    }
  }

  private loadAndDisplayText() {
    const textLabels = Text.loadAndDisplayText();
    for (const label of textLabels) {
      this.group.add(label);
    }
  }
}
