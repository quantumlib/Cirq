import {createCanvas} from 'canvas';

function installNodeCanvas2D(doc: Document | undefined | null) {
  const win = (doc as any)?.defaultView as Window | undefined;
  const CanvasCtor = win?.HTMLCanvasElement;
  if (!CanvasCtor) return;
  const proto = CanvasCtor.prototype as any;
  Object.defineProperty(proto, 'getContext', {
    configurable: true,
    writable: true,
    value: function getContext(type: string) {
      if (type !== '2d') return null;
      const width = (this as HTMLCanvasElement).width || 300;
      const height = (this as HTMLCanvasElement).height || 150;
      const backing = createCanvas(width, height);
      return backing.getContext('2d') as unknown as CanvasRenderingContext2D;
    },
  });
}

// Apply to current document
installNodeCanvas2D(globalThis.document);

// Re-apply when tests replace global document
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
