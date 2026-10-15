// Reads a video's frames in order for processing.
// Fast path: demux with web-demuxer and decode with WebCodecs, so frames stream straight out of the
// decoder instead of seeking to each one (seeking a 4K video re-decodes from the last keyframe
// every time). Fallback: seek the <video> element frame by frame, like before.
import { WebDemuxer } from '../vendor/web-demuxer/web-demuxer.js';

const WASM = new URL('../vendor/web-demuxer/web-demuxer.wasm', import.meta.url).href;

function abortError() { return new DOMException('취소됨', 'AbortError'); }

export function seekVideo(video, time) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('영상 탐색 시간이 초과됐어요.')); }, 10000);
    const onSeeked = () => { cleanup(); resolve(); };
    const cleanup = () => { clearTimeout(timer); video.removeEventListener('seeked', onSeeked); };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}

/**
 * Opens `file` for sequential reading. `video` is the same file loaded in a <video>, used for its
 * display size (which already accounts for rotation) and as the fallback reader.
 */
export async function openSource(file, video) {
  try {
    if (typeof VideoDecoder === 'undefined') throw new Error('no WebCodecs');
    const demuxer = new WebDemuxer({ wasmFilePath: WASM });
    await demuxer.load(file);
    const info = await demuxer.getMediaInfo();
    const vstream = info.streams.find(s => s.codec_type_string === 'video' || s.codec_type === 0);
    const config = await demuxer.getDecoderConfig('video');
    const support = await VideoDecoder.isConfigSupported(config);
    if (!support.supported) throw new Error('decoder unsupported: ' + config.codec);
    let audio = null;
    try {
      const aconf = await demuxer.getDecoderConfig('audio');
      const mux = /^mp4a/.test(aconf.codec) ? 'aac' : aconf.codec === 'opus' ? 'opus' : null;
      if (mux) audio = { config: aconf, mux };
    } catch { /* no audio track */ }
    return new DecoderSource(demuxer, config, video, Number(vstream?.rotation) || 0, audio);
  } catch (err) {
    console.warn('Sequential decoding unavailable, falling back to seeking:', err);
    return new SeekSource(video);
  }
}

// Draws `img` (a VideoFrame or the <video>) upright: phone videos store frames sideways
// with a rotation flag that a VideoFrame doesn't apply by itself.
function drawUpright(ctx, img, rotation, dispW, dispH, sx, sy, sw, sh, dw, dh) {
  const fw = img.displayWidth ?? img.videoWidth, fh = img.displayHeight ?? img.videoHeight;
  const needsTurn = rotation % 180 !== 0 && fw === dispH && fh === dispW && fw !== fh;
  if (!needsTurn) { ctx.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh); return; }
  // Draw the whole frame rotated into display space, scaled so the source rect lands on (0,0,dw,dh).
  const k = dw / sw, kh = dh / sh;
  ctx.save();
  ctx.scale(k, kh);
  ctx.translate(-sx, -sy);
  const r = ((rotation % 360) + 360) % 360;
  // rotation is clockwise as stored by the container (e.g. 90 for iPhone portrait).
  if (r === 90) { ctx.translate(dispW, 0); ctx.rotate(Math.PI / 2); }
  else if (r === 270) { ctx.translate(0, dispH); ctx.rotate(-Math.PI / 2); }
  ctx.drawImage(img, 0, 0, fw, fh);
  ctx.restore();
}

class DecoderSource {
  constructor(demuxer, config, video, rotation, audio) {
    this.kind = 'decoder';
    this.demuxer = demuxer;
    this.config = config;
    this.video = video;
    this.rotation = rotation;
    this.audio = audio; // { config, mux } when the audio can be copied as-is
  }

  /**
   * Yields one frame per output time (start + i/fps). The frame is valid until the next step.
   * Each yielded item has draw(ctx, rect, dw, dh).
   */
  async *frames(start, end, fps, signal) {
    const dispW = this.video.videoWidth, dispH = this.video.videoHeight;
    const queue = [];
    let wake = null, failure = null, eof = false;
    const notify = () => { if (wake) { const w = wake; wake = null; w(); } };
    const decoder = new VideoDecoder({
      output: f => { queue.push(f); notify(); },
      error: e => { failure = e; notify(); },
    });
    decoder.configure(this.config);
    decoder.addEventListener?.('dequeue', notify);
    // No end bound: the loop below stops once it has every output frame. (A track can start later
    // than the file's timeline; the <video> element shows its first frame until then, and so do we.)
    const reader = this.demuxer.read('video', Math.max(0, start)).getReader();

    // Next decoded frame, feeding the decoder only as fast as we use frames (decoders stall
    // when too many frames are held).
    const nextFrame = async () => {
      for (;;) {
        if (signal?.aborted) throw abortError();
        if (failure) throw failure;
        if (queue.length) return queue.shift();
        if (eof) return null;
        if (decoder.decodeQueueSize > 6) { await new Promise(r => { wake = r; setTimeout(r, 50); }); continue; }
        const { value, done } = await reader.read();
        if (done) {
          await decoder.flush().catch(e => { failure = e; });
          eof = true;
          continue;
        }
        decoder.decode(value);
        if (!queue.length) await new Promise(r => { wake = r; setTimeout(r, 5); });
      }
    };

    let cur = null, next = await nextFrame();
    const total = Math.max(1, Math.floor((end - start) * fps));
    try {
      for (let i = 0; i < total; i++) {
        const t = (start + (i + 0.5) / fps) * 1e6;
        // Advance to the last frame that starts at or before t.
        while (next && (next.timestamp <= t || !cur)) {
          cur?.close();
          cur = next;
          next = await nextFrame();
        }
        if (!cur) break;
        const frame = cur;
        yield {
          index: i, total,
          draw: (ctx, r, dw, dh) => drawUpright(ctx, frame, this.rotation, dispW, dispH, r.x, r.y, r.w, r.h, dw, dh),
        };
      }
    } finally {
      cur?.close();
      next?.close();
      for (const f of queue) f.close();
      reader.cancel().catch(() => {});
      if (decoder.state !== 'closed') decoder.close();
    }
  }

  // Encoded audio chunks between start and end, re-timed to begin at 0.
  async *audioChunks(start, end, signal) {
    const reader = this.demuxer.read('audio', Math.max(0, start), end).getReader();
    try {
      for (;;) {
        if (signal?.aborted) throw abortError();
        const { value, done } = await reader.read();
        if (done) return;
        const ts = value.timestamp - start * 1e6;
        if (ts < -1e4) continue;
        if (ts > (end - start) * 1e6) return;
        const data = new Uint8Array(value.byteLength);
        value.copyTo(data);
        yield new EncodedAudioChunk({ type: value.type, timestamp: Math.max(0, Math.round(ts)), duration: value.duration ?? undefined, data });
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  close() { try { this.demuxer.destroy(); } catch {} }
}

class SeekSource {
  constructor(video) {
    this.kind = 'seek';
    this.video = video;
    this.audio = null;
  }

  async *frames(start, end, fps, signal) {
    const v = this.video;
    const total = Math.max(1, Math.floor((end - start) * fps));
    for (let i = 0; i < total; i++) {
      if (signal?.aborted) throw abortError();
      await seekVideo(v, Math.min(start + (i + 0.5) / fps, v.duration - 0.001));
      yield { index: i, total, draw: (ctx, r, dw, dh) => ctx.drawImage(v, r.x, r.y, r.w, r.h, 0, 0, dw, dh) };
    }
  }

  close() {}
}
