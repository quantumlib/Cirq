import { Line } from 'three';
export declare class Meridians {
    private static _curveData;
    /**
     * Creates the special horizontal meridian lines of the bloch
     * sphere, each with a different radius and location.
     * @param radius The radius of the overall bloch sphere
     * @returns A list of circles (reprsented as Line objs) to draw on the scene
     */
    static createHorizontalChordMeridians(radius: number): Line[];
    /**
     * Creates equally sized horizontal meridian lines which rotate
     * by varying degrees across the same axis.
     * @param radius The radius of the overall bloch sphere
     * @returns A list of circles (represented as Line objs) to draw on the scene
     */
    static createHorizontalCircleMeridians(radius: number): Line[];
    /** Creates equally sized vertical meridian lines which rotate
     * by varying degrees across the same axis
     * @param radius The radius of the overall bloch sphere
     * @returns A list of circles (represented as Line objs) to draw on the scene
     */
    static createVerticalMeridians(radius: number): Line[];
    /**
     * Helper function that generates a necessary EllipseCurve
     * given the required information.
     * @param curveData An object that contains info about the curve
     * @returns An EllipseCurve object based off the curve information.
     */
    private static _createMeridianCurve;
    /**
     * Helper function that generates the actual Line object which will be
     * rendered by the three.js scene.
     * @param curve An EllipseCurve object that provides location/size info
     * @param rotationFactor The desired angle of rotation in radians
     * @param vertical (Optional) boolean that tells whether or not we're generating a horizontal
     * or vertical line.
     * @param yPosition (Optional) Allows the yPosition of the line to be updated to
     * the provided value
     * @returns A Line object that can be rendered by a three.js scene.
     */
    private static _createMeridianLine;
}
