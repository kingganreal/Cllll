// One character on the desktop. Only her visible pixels catch the mouse; everywhere else
// clicks fall through to whatever is behind the window.
import { AlphaClip } from './player.js';

const id = new URLSearchParams(location.search).get('id');
const info = await window.fancam.getOverlay(id);
if (!info) window.close();

document.title = info.clip.name;
const player = new AlphaClip(info.clip, { readable: true });
document.body.append(player.canvas);

function apply(s) {
  player.setFlip(!!s.flip);
  player.video.playbackRate = s.speed || 1;
  player.video.muted = !!s.muted;
  player.play();
}
apply(info.s);
window.fancam.onState(apply);

const fit = () => player.resize(innerWidth, innerHeight);
addEventListener('resize', fit);
fit();

let ignoring = true, dragging = false;
function setIgnore(v) {
  if (v === ignoring) return;
  ignoring = v;
  window.fancam.setIgnoreMouse(v);
}

addEventListener('mousemove', e => {
  if (dragging) return;
  setIgnore(player.alphaAt(e.clientX, e.clientY) < 40);
});
addEventListener('mouseleave', () => { if (!dragging) setIgnore(true); });
addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  dragging = true;
  window.fancam.drag(true, e.clientX, e.clientY);
});
addEventListener('mouseup', e => {
  if (e.button !== 0 || !dragging) return;
  dragging = false;
  window.fancam.drag(false);
});
addEventListener('wheel', e => { window.fancam.zoom(Math.exp(-e.deltaY * 0.0015)); }, { passive: true });
addEventListener('contextmenu', e => { e.preventDefault(); window.fancam.menu(); });
