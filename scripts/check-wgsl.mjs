// A backtick inside a WGSL template literal ends the literal.
//
// The rest of the string then parses as TypeScript, so tsc reports a syntax
// error several lines away from the comment that caused it, pointing at code
// that looks perfectly fine. It cost five round trips in a single session before
// anyone wrote this down, which is what a check is for.
//
// The WGSL in this project lives in `/* wgsl */ \`...\`` template literals. This
// walks those regions and flags any backtick that is not escaped. Escaped ones
// (\\\`) are deliberate and common -- the shader comments quote identifiers.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name.startsWith('.')) continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (p.endsWith('.ts')) yield p;
  }
}

const MARKER = '/* wgsl */ `';
let bad = 0;

for (const file of walk('src')) {
  const src = readFileSync(file, 'utf8');
  let at = 0;
  while ((at = src.indexOf(MARKER, at)) !== -1) {
    const open = at + MARKER.length;
    // Scan to the closing delimiter, honouring escapes. An UNescaped backtick is
    // the close -- so if it lands mid-comment, that comment is the bug.
    let i = open;
    for (; i < src.length; i++) {
      if (src[i] === '\\') { i++; continue; }
      if (src[i] === '`') break;
    }
    // The literal is well-formed only if what follows the close is `;` or `+`.
    // Anything else means the close was premature, i.e. a stray backtick.
    const after = src.slice(i + 1).match(/^\s*([;+])/);
    if (!after) {
      const line = src.slice(0, i).split('\n').length;
      const text = src.split('\n')[line - 1].trim();
      console.error(`${file}:${line}  stray backtick inside a WGSL literal\n    ${text}`);
      bad++;
    }
    at = i + 1;
  }
}

if (bad > 0) {
  console.error(`\n${bad} stray backtick(s). Inside /* wgsl */ literals, quote identifiers with "double quotes" or escape as \\\`.`);
  process.exit(1);
}
console.log('✓ WGSL literals — no stray backticks');
