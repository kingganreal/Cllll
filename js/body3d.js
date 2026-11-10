// 3D clips (.fancam3d, made on Colab by colab/body3d.py): her body as a moving mesh (SAM 3D Body),
// played with its song in a three.js view you can turn and zoom.
import * as THREE from '../vendor/three/three.module.min.js';
import { OrbitControls } from '../vendor/three/OrbitControls.js';

/** Reads a .fancam3d file: { header, faces, frames (Int16Array), audio (Blob|null) }. */
export async function parse(blob) {
  const buf = await blob.arrayBuffer();
  const dv = new DataView(buf);
  if (new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== 'F3D1') throw new Error('3D 클립 파일이 아니에요.');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, dv.getUint32(4, true))));
  const faces = header.faceType === 'u16'
    ? new Uint16Array(buf, header.facesOffset, header.faceCount * 3)
    : new Uint32Array(buf, header.facesOffset, header.faceCount * 3);
  const frames = new Int16Array(buf, header.framesOffset, header.frames * header.vertices * 3);
  const audio = header.audioLength ? new Blob([new Uint8Array(buf, header.audioOffset, header.audioLength)], { type: header.audioType }) : null;
  return { header, faces, frames, audio };
}

export class Body3D {
  /** @param {HTMLElement} host where the canvas goes */
  constructor(host, data) {
    this.data = data;
    const { header, faces } = data;
    this.n = header.vertices;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    host.append(this.renderer.domElement);
    this.host = host;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
    this.camera.position.set(0, 1.0, 4.2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.9, 0);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xfff4fa, 0x40324a, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 3, 2.5);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xc9a7ff, 0.8);
    rim.position.set(-2, 2, -2.5);
    this.scene.add(rim);

    // A soft shadow under her feet.
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const g = c.getContext('2d'), grad = g.createRadialGradient(64, 64, 4, 64, 64, 64);
    grad.addColorStop(0, 'rgba(0,0,0,.45)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    this.shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2;
    this.scene.add(this.shadow);

    this.positions = new Float32Array(this.n * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    this.mesh = new THREE.Mesh(this.geometry, new THREE.MeshStandardMaterial({ color: 0xf3e7ee, roughness: 0.55, metalness: 0.05 }));
    this.scene.add(this.mesh);

    this.audio = data.audio ? new Audio(URL.createObjectURL(data.audio)) : null;
    if (this.audio) this.audio.preload = 'auto';
    this.duration = header.frames / header.fps;
    this.clock = 0; this.playing = false; this.last = performance.now();
    this.show(0);
    this.resize();
    this.onResize = () => this.resize();
    addEventListener('resize', this.onResize);
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
  }

  resize() {
    const w = this.host.clientWidth || 640, h = this.host.clientHeight || 480;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** The pose at time t (seconds), blended between the two nearest frames. */
  show(t) {
    const { header, frames } = this.data, n3 = this.n * 3;
    const f = Math.max(0, Math.min(header.frames - 1, t * header.fps));
    const a = Math.floor(f), b = Math.min(header.frames - 1, a + 1), w = f - a;
    const [sx, sy, sz] = header.scale, [ox, oy, oz] = header.offset;
    const s = [sx, sy, sz], o = [ox, oy, oz];
    const A = a * n3, B = b * n3, p = this.positions;
    for (let k = 0; k < n3; k++) {
      const c = k % 3;
      p[k] = o[c] + (frames[A + k] * (1 - w) + frames[B + k] * w) * s[c];
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.geometry.computeBoundingSphere();
    // the shadow follows her feet
    let x = 0, z = 0;
    for (let k = 0; k < n3; k += 30) { x += p[k]; z += p[k + 2]; }
    const m = Math.ceil(this.n / 10);
    this.shadow.position.set(x / m, 0.002, z / m);
  }

  play() {
    this.playing = true; this.last = performance.now();
    if (this.audio) { this.audio.currentTime = this.clock; this.audio.play().catch(() => {}); }
  }
  pause() { this.playing = false; this.audio?.pause(); }
  toggle() { this.playing ? this.pause() : this.play(); return this.playing; }

  loop(now) {
    if (this.playing) {
      // the song is the clock when there is one, so the dance stays on the beat
      this.clock = this.audio && !this.audio.paused ? this.audio.currentTime : this.clock + (now - this.last) / 1000;
      if (this.clock >= this.duration) {
        this.clock = 0;
        if (this.audio) { this.audio.currentTime = 0; this.audio.play().catch(() => {}); }
      }
      this.show(this.clock);
    }
    this.last = now;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.raf = requestAnimationFrame(this.loop);
  }

  /** A PNG of the current view (for the library card). */
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return new Promise(r => this.renderer.domElement.toBlob(r, 'image/png'));
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    removeEventListener('resize', this.onResize);
    this.pause();
    if (this.audio) URL.revokeObjectURL(this.audio.src);
    this.controls.dispose();
    this.geometry.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

/** A thumbnail for a 3D clip: its middle frame, rendered off screen. */
export async function thumbnail(blob) {
  const host = document.createElement('div');
  host.style.cssText = 'position:fixed;left:-9999px;width:360px;height:360px';
  document.body.append(host);
  const view = new Body3D(host, await parse(blob));
  try {
    view.show(view.duration / 2);
    return await view.snapshot();
  } finally {
    view.dispose();
    host.remove();
  }
}
