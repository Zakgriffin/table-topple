// ── Tiny shared SVG element helpers ───────────────────────────────────────
//
// Used by every overlay that draws into a real <svg> element --
// gridPeriodPhaseOverlays.ts's own plot, overlays/lsdOverlay.ts's
// rectangle/composite-line drawing, and overlays/hoverDebugOverlays.ts's
// gradient/level-line arrows.

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

// Shared arrow draw -- used by hoverDebugOverlays.ts's gradient/level-line
// arrows and lsdOverlay.ts's growth-candidate preview. Black halo drawn as
// the FIRST three children, colored arrow as the next three -- SVG paints
// later siblings on top of earlier ones, so this ordering is what keeps the
// halo underneath.
export function drawOneArrow(group: SVGGElement, px: number, py: number, dirVecX: number, dirVecY: number, color: string, scale: number) {
  const tipX = px + dirVecX * scale, tipY = py + dirVecY * scale;
  const headLen = 8, headAngle = Math.PI / 7;
  const backAngle = Math.atan2(tipY - py, tipX - px);
  const hx1 = tipX - headLen * Math.cos(backAngle - headAngle), hy1 = tipY - headLen * Math.sin(backAngle - headAngle);
  const hx2 = tipX - headLen * Math.cos(backAngle + headAngle), hy2 = tipY - headLen * Math.sin(backAngle + headAngle);
  const headPoints = `${tipX},${tipY} ${hx1},${hy1} ${hx2},${hy2}`;
  // Shaft stops at the arrowhead's own back edge (the midpoint of hx1/hy1
  // and hx2/hy2, at distance headLen*cos(headAngle) back from the tip along
  // the shaft), not at the tip itself -- otherwise the shaft's stroke width
  // pokes out past the arrowhead polygon's own edges right near the point,
  // since the triangle narrows to zero width exactly where the shaft ends.
  const backEdgeDist = headLen * Math.cos(headAngle);
  const shaftEndX = tipX - backEdgeDist * Math.cos(backAngle), shaftEndY = tipY - backEdgeDist * Math.sin(backAngle);

  group.appendChild(svgEl('line', { x1: px, y1: py, x2: shaftEndX, y2: shaftEndY, stroke: 'rgba(0,0,0,0.85)', 'stroke-width': 4 }));
  group.appendChild(svgEl('polygon', { points: headPoints, fill: 'rgba(0,0,0,0.85)' }));
  group.appendChild(svgEl('circle', { cx: px, cy: py, r: 3.5, fill: 'rgba(0,0,0,0.85)' }));

  group.appendChild(svgEl('line', { x1: px, y1: py, x2: shaftEndX, y2: shaftEndY, stroke: color, 'stroke-width': 2 }));
  group.appendChild(svgEl('polygon', { points: headPoints, fill: color }));
  group.appendChild(svgEl('circle', { cx: px, cy: py, r: 2.5, fill: color }));
}
