import {createSphere} from './components/sphere';
import {generateAxis} from './components/axes';
import {
  createHorizontalChordMeridians,
  createVerticalMeridians,
} from './components/meridians';
import {loadAndDisplayText} from './components/text';

import {Group} from 'three';

/**
 * Class bringinging together the individual components like the
 * Sphere, Axes, Meridicans, and Text into the overall visualization
 * of the Bloch sphere.
 */
export class BlochSphere {
  private radius: number;
  private group: Group;

  constructor(radius = 5) {
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
  getBlochSphere(): Group {
    // Return a clone of the group to avoid mutation.
    return this.group.clone();
  }

  /**
   * Returns the radius of the bloch_sphere.
   * Used for testing purposes.
   */
  getRadius() : number {
    const radius = this.radius;
    return radius;
  }

  private add3dSphere() {
    const sphere = createSphere(this.radius);
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
    const sprites = loadAndDisplayText();
    for (const sprite of sprites){
      this.group.add(sprite);
    }
  }
}
