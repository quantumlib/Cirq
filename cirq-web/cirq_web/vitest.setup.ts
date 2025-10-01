import {createCanvas} from 'canvas';

/**
 * This is a workaround for the fact that canvas is not supported in jsdom,
 * which does not allow for calling methods like canvas.getContext(), which
 * is used in three.js components. 
 * 
 * We implement this working using a lightweight, test-only
 * polyfill (https://developer.mozilla.org/en-US/docs/Glossary/Polyfill).
 * Node-canvas is used as the backend drawing "surface", overriding
 * the <canvas> that we'd really like.
 * 
 * See a Github convesation about this
 * https://github.com/vitest-dev/vitest/issues/274
 * 
 * Rather than relying on an external package and its dependencies, 
 * we offer a simple implementation for our own use cases.
 */
function installNodeCanvas2D(doc: Document | undefined | null) {
  // Sets up the Window
  const win = (doc as any)?.defaultView as Window | undefined;
  const CanvasCtor = win?.HTMLCanvasElement;
  if (!CanvasCtor) return;


  const proto = CanvasCtor.prototype as any;
  Object.defineProperty(proto, 'getContext', {
    configurable: true,
    writable: true,
    // Overwrites getContext, as mentioned above.
    value: function getContext(type: string) {
      if (type !== '2d') return null;
      // Mocks the default width and height of the canvas as
      // specified by the HTML standard:
      // https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/canvas
      const width = (this as HTMLCanvasElement).width || 300;
      const height = (this as HTMLCanvasElement).height || 150;

      const backing = createCanvas(width, height);
      return backing.getContext('2d') as unknown as CanvasRenderingContext2D;
    },
  });
}

// Apply to current document
installNodeCanvas2D(globalThis.document);

/**
 * This is Vitest specific, as it can swap out the globalThis.document
 * when moving between workers, so this makes sure the polyfill is reapplied
 * and available for every test.
 */
let _doc: Document = globalThis.document;
Object.defineProperty(globalThis, 'document', {
  configurable: true,
  get() {
    return _doc;
  },
  set(value: Document) {
    _doc = value;
    try {
      installNodeCanvas2D(value);
    } catch {}
  },
});
