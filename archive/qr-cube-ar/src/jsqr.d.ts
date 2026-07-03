declare module 'jsqr' {
  interface Point { x: number; y: number; }

  interface QRLocation {
    topLeftCorner: Point;
    topRightCorner: Point;
    bottomLeftCorner: Point;
    bottomRightCorner: Point;
    topLeftFinderPattern: Point;
    topRightFinderPattern: Point;
    bottomLeftFinderPattern: Point;
  }

  interface QRCode {
    binaryData: number[];
    data: string;
    chunks: unknown[];
    version: number;
    location: QRLocation;
  }

  interface Options {
    inversionAttempts?: 'dontInvert' | 'onlyInvert' | 'attemptBoth' | 'invertFirst';
  }

  function jsQR(data: Uint8ClampedArray, width: number, height: number, options?: Options): QRCode | null;
  export default jsQR;
}
