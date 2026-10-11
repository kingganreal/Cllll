// Plays an alpha-packed clip (see processor.js) as a transparent, animated sprite.
const VERT = `
attribute vec2 p;
varying vec2 v;
void main() { v = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision mediump float;
varying vec2 v;
uniform sampler2D tex;
uniform vec4 crop;     // x, y, w, h inside one half (0..1)
uniform float layout;  // 0 = side-by-side, 1 = top/bottom
uniform float flip;
uniform vec2 texel;    // half a texel, keeps samples away from the seam
void main() {
  vec2 uv = v;
  if (flip > 0.5) uv.x = 1.0 - uv.x;
  uv = crop.xy + uv * crop.zw;
  vec2 cu, au;
  if (layout < 0.5) {
    uv.x = clamp(uv.x * 0.5, texel.x, 0.5 - texel.x);
    cu = uv; au = vec2(uv.x + 0.5, uv.y);
  } else {
    uv.y = clamp(uv.y * 0.5, texel.y, 0.5 - texel.y);
    cu = uv; au = vec2(uv.x, uv.y + 0.5);
  }
  vec3 c = texture2D(tex, cu).rgb;
  float a = texture2D(tex, au).r;
  a = smoothstep(0.04, 0.96, a);
  gl_FragColor = vec4(c * a, a);
}`;

export class AlphaClip {
  /** @param {{blob: Blob, layout: string, crop: object, width: number, height: number}} clip */
  constructor(clip) {
    this.clip = clip;
    this.flip = false;
    this.url = URL.createObjectURL(clip.blob);

    const video = this.video = document.createElement('video');
    video.src = this.url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('webkit-playsinline', '');
    video.preload = 'auto';

    const canvas = this.canvas = document.createElement('canvas');
    canvas.className = 'clip-canvas';
    const gl = this.gl = canvas.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: false })
      || canvas.getContext('experimental-webgl');
    if (!gl) throw new Error('WebGL을 쓸 수 없는 브라우저예요.');
    this.#initGL();

    this.aspect = (clip.width * clip.crop.w) / (clip.height * clip.crop.h);
    this.#loop();
  }

  #initGL() {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
      [gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    this.u = {
      crop: gl.getUniformLocation(prog, 'crop'),
      layout: gl.getUniformLocation(prog, 'layout'),
      flip: gl.getUniformLocation(prog, 'flip'),
      texel: gl.getUniformLocation(prog, 'texel'),
    };
    const { crop, layout, width, height } = this.clip;
    gl.uniform4f(this.u.crop, crop.x, crop.y, crop.w, crop.h);
    gl.uniform1f(this.u.layout, layout === 'tb' ? 1 : 0);
    const W = layout === 'sbs' ? width * 2 : width, H = layout === 'sbs' ? height : height * 2;
    gl.uniform2f(this.u.texel, 0.5 / W, 0.5 / H);
    gl.clearColor(0, 0, 0, 0);
  }

  #loop() {
    const step = () => {
      if (this.destroyed) return;
      this.draw();
      if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(step);
      else requestAnimationFrame(step);
    };
    if (this.video.requestVideoFrameCallback) {
      this.video.requestVideoFrameCallback(step);
      this.video.addEventListener('seeked', () => this.draw());
    } else {
      requestAnimationFrame(step);
    }
  }

  // Match the canvas backing store to its on-screen size (capped to the source pixels).
  resize(cssWidth, cssHeight) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const maxW = this.clip.width * this.clip.crop.w * 1.5;
    const s = Math.min(dpr, maxW / Math.max(1, cssWidth));
    const w = Math.max(1, Math.round(cssWidth * s)), h = Math.max(1, Math.round(cssHeight * s));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
      this.draw();
    }
  }

  draw() {
    const { gl, video } = this;
    if (video.readyState < 2) return;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    gl.uniform1f(this.u.flip, this.flip ? 1 : 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  setFlip(f) { this.flip = f; this.draw(); }

  play() { return this.video.play().catch(() => {}); }

  destroy() {
    this.destroyed = true;
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    URL.revokeObjectURL(this.url);
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
