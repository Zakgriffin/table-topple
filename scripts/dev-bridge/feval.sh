#!/bin/bash
# Focus-gated dev-bridge eval -- use this instead of `cli.js eval` for ANYTHING
# whose timing matters.
#
# Accepting a command in the terminal takes focus AWAY from the browser tab, and
# an unfocused tab throttles rAF and timers -- so a measurement kicked off the
# instant a command is accepted is wrong, not merely noisy. This polls until the
# page reports focus again, lets it settle, and only then runs the payload. The
# whole thing is ONE accepted command, so the poll runs after the accept-click
# has already given focus back.
#
#   scripts/dev-bridge/feval.sh "<js>"             run once focused
#   SETTLE=3 scripts/dev-bridge/feval.sh "<js>"     longer settle before running
#
# ── Driving a measurement from cold, which took several tries to get right ──
#
# cli.js's eval handler runs `eval(code)` SYNCHRONOUSLY and replies with whatever
# the expression returns -- it does NOT await promises. So anything async has to
# park its result on a global and be polled:
#
#   feval.sh "globalThis.__r=null; timeReconstruction(globalThis.__cam, 9)
#             .then(r=>{globalThis.__r=r}).catch(e=>{globalThis.__e=String(e)}); 'started'"
#   sleep 18
#   feval.sh "globalThis.__e || (globalThis.__r ? formatReconstructionTiming(globalThis.__r) : 'pending')"
#
# And there is NO DEFAULT CAMERA -- after any source edit the vite page reloads,
# `cameras` is empty and the capture is gone. Full cold start:
#
#   feval.sh "if (cameras.size===0) addSimulatedCamera();
#             var c=[...cameras.values()][0]; globalThis.__cam=c;
#             setActiveCameraId(c.id); renderCameraTabs(); refreshCameraPanel();
#             resizeCaptureBuffers(c); markCaptureDirty(c);
#             runAxesReconstruction(c); 'capture started'"
#   sleep 5   # runAxesReconstruction defers its real work via requestAnimationFrame
set -euo pipefail
cd /Users/zakgriffin/Desktop/cube
SETTLE="${SETTLE:-0.8}"
DEADLINE=$((SECONDS + 120))

while :; do
  got=$(node scripts/dev-bridge/cli.js eval "document.hasFocus() && document.visibilityState==='visible'" 2>/dev/null | tr -d ' \n' | grep -o '"value":true' || true)
  [ -n "$got" ] && break
  if [ $SECONDS -ge $DEADLINE ]; then echo "feval: page never regained focus" >&2; exit 1; fi
  sleep 0.5
done
sleep "$SETTLE"

node scripts/dev-bridge/cli.js eval "$1" 2>&1 | python3 -c "
import sys,json
raw=sys.stdin.read()
try:
    o=json.loads(raw[raw.index('{'):])
except Exception:
    print(raw); raise SystemExit(0)
if not o.get('ok'): print('ERROR:', o.get('error')); raise SystemExit(1)
v=o.get('value')
print(v if isinstance(v,str) else json.dumps(v,indent=1))
"
