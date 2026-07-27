// ── Tiny shared SVG element helpers ───────────────────────────────────────
//
// Used by any overlay that draws into a real <svg> element (as opposed to
// the shared gradientArrowCanvas's 2D canvas draws) -- gridPeriodPhaseOverlays.ts's
// own plot, and overlays/lsdOverlay.ts's rectangle/composite-line drawing.

export const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag) as SVGElementTagNameMap[K];
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function svgText(x: number, y: number, content: string, attrs: Record<string, string | number>): SVGTextElement {
  const el = svgEl('text', { x, y, ...attrs });
  el.textContent = content;
  return el;
}
