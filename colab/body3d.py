"""직캠 3D: her body as a moving 3D mesh, for the app's 3D viewer.

Her box is followed with YOLO and her face (fancam.YoloFollower, as in circle mode), and Meta's
SAM 3D Body turns her in each frame into a body mesh (Momentum Human Rig: pose, shape, hands),
placed where the camera saw her. The meshes are smoothed over time and written with the song as
one .fancam3d file:

  'F3D1' | u32 header length | header JSON (utf-8, padded to 4 bytes) | faces | frames | audio
  header: {name, fps, frames, vertices, faceCount, faceType: 'u16'|'u32', scale: [3], offset: [3],
           facesOffset, framesOffset, audioOffset, audioLength, audioType}
  frames: int16 x, y, z per vertex per frame; position = offset + q * scale (y up, metres)

The checkpoints are gated: request access to facebook/sam-3d-body-dinov3 on Hugging Face and log
in (the notebook asks for a token). Code: https://github.com/facebookresearch/sam-3d-body (SAM
License).
"""
import json
import os
import struct
import subprocess
import sys

import cv2
import numpy as np

import fancam
from fancam import CACHE, Faces, Frames, Matcher, Stopped, YoloFollower, probe, reference_from_photos

REPO_URL = 'https://github.com/facebookresearch/sam-3d-body.git'
REPO = '/content/sam-3d-body' if os.path.isdir('/content') else os.path.join(CACHE, 'sam-3d-body')

_estimator = None


def estimator(hf_repo='facebook/sam-3d-body-dinov3'):
    """SAM 3D Body, set up once (clones its code; the checkpoint comes from Hugging Face)."""
    global _estimator
    if _estimator is None:
        if not os.path.isdir(REPO):
            subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO], check=True)
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from notebook.utils import setup_sam_3d_body
        # No detector: her box comes from the YOLO follower (and it spares installing detectron2).
        # MoGe-2 estimates the camera's field of view; without it a default one is used.
        try:
            _estimator = setup_sam_3d_body(hf_repo_id=hf_repo, detector_name='', fov_name='moge2')
        except ImportError as err:
            print('MoGe 없이 진행해요 (기본 화각):', err)
            _estimator = setup_sam_3d_body(hf_repo_id=hf_repo, detector_name='', fov_name='')
    return _estimator


def smooth(frames, keep=0.5):
    """Calms the per-frame jitter of single-image estimates: each frame leans on the last one,
    except where the body really moved a lot (a fast spin isn't smeared)."""
    out = [frames[0]]
    for v in frames[1:]:
        prev = out[-1]
        moved = np.linalg.norm(v - prev, axis=1, keepdims=True)
        w = np.clip(moved / 0.05, 1 - keep, 1)  # 5 cm or more: take the new position as is
        out.append(prev + w * (v - prev))
    return out


def write_file(path, name, fps, frames, faces, audio=None):
    V = frames[0].shape[0]
    allv = np.stack(frames)                       # (T, V, 3)
    lo, hi = allv.reshape(-1, 3).min(0), allv.reshape(-1, 3).max(0)
    offset = (lo + hi) / 2
    scale = np.maximum((hi - lo) / 2 / 32767, 1e-6)
    q = np.round((allv - offset) / scale).clip(-32767, 32767).astype('<i2')
    face_type = 'u16' if V < 65536 else 'u32'
    face_bytes = np.asarray(faces, '<u2' if face_type == 'u16' else '<u4').tobytes()
    header = {'name': name, 'fps': fps, 'frames': len(frames), 'vertices': V, 'faceCount': len(faces),
              'faceType': face_type, 'scale': scale.tolist(), 'offset': offset.tolist(),
              'audioType': 'audio/mp4', 'audioLength': len(audio or b'')}
    # offsets depend on the header's own length: settle them in two passes
    for _ in range(2):
        hb = json.dumps(header).encode()
        hb += b' ' * (-(8 + len(hb)) % 4)
        start = 8 + len(hb)
        header['facesOffset'] = start
        header['framesOffset'] = start + len(face_bytes) + (-len(face_bytes) % 4)
        header['audioOffset'] = header['framesOffset'] + q.nbytes
    hb = json.dumps(header).encode()
    hb += b' ' * (-(8 + len(hb)) % 4)
    assert 8 + len(hb) == header['facesOffset']
    with open(path, 'wb') as f:
        f.write(b'F3D1'); f.write(struct.pack('<I', len(hb))); f.write(hb)
        f.write(face_bytes); f.write(b'\0' * (-len(face_bytes) % 4))
        f.write(q.tobytes())
        if audio:
            f.write(audio)


def make_3d(video, out_dir, name='설윤', photos=(), start=0.0, end=None, fps=15, max_side=1280,
            ref=None, pick=None, on_progress=None, should_stop=None, cancelled=None, face_every=0.25):
    """Returns {'path', 'frames', 'duration', 'fixes', 'partial'}; arguments as fancam.make_clip."""
    report = on_progress or (lambda done, total, stage=None: None)
    vw, vh, dur = probe(video)
    end = min(end or dur, dur)
    s = max_side / max(vw, vh)
    W, H = fancam.even(vw * s), fancam.even(vh * s)
    frames = Frames(video, start, end, fps, (W, H))
    total = frames.total
    report(0, total, '3D 모델 준비 중 (처음엔 몇 분 걸려요)')

    faces = Faces()
    if ref is not None:
        ref = np.asarray(ref, np.float32); ref /= np.linalg.norm(ref)
    elif photos:
        ref = reference_from_photos(faces, photos)
    if ref is None and pick is None:
        ref = fancam.reference_from_video(faces, video)
    her_face = Matcher(ref) if ref is not None else None

    def look(bgr):
        found = faces(bgr)
        if her_face is None:
            return found, None
        middle = lambda f: 0.2 < (f['box'][0] + f['box'][2]) / 2 / W < 0.8
        return found, her_face.find(found, lambda f: middle(f) or f['sim'] >= 0.55)

    first_her = None
    if pick is None:
        found, first_her = look(frames.get(0))
        if first_her is None:
            raise RuntimeError('첫 장면에서 설윤 얼굴을 찾지 못했어요. 앱에서 멤버를 골라 보내 주세요.')
    follower = YoloFollower(fps, max(1, int(round(fps * face_every))), her_face, look, pick, first_her)
    est = estimator()

    meshes, last = [], None
    sx = vw / W
    for i in range(total):
        if cancelled and cancelled():
            raise Stopped('취소됐어요')
        if should_stop and should_stop() and meshes:
            break
        box = follower.step(i, frames.get(i))
        v = None
        if box is not None:
            rgb = cv2.cvtColor(frames.original(i), cv2.COLOR_BGR2RGB)
            b = np.array([[box[0] * sx, box[1] * sx, box[2] * sx, box[3] * sx]], np.float32)
            outs = est.process_one_image(rgb, bboxes=b)
            if outs:
                o = outs[0]
                v = np.asarray(o['pred_vertices'], np.float32) + np.asarray(o['pred_cam_t'], np.float32)
                v = v * np.array([1, -1, -1], np.float32)  # camera (y down, z ahead) -> y up, facing us
        if v is None:
            v = last  # missed her this frame: hold the last pose
        if v is not None:
            meshes.append(v)
            last = v
        report(i + 1, total, None)
        frames.forget_before(i)
    frames.close()
    if not meshes:
        raise RuntimeError('3D로 만들 장면을 찾지 못했어요.')
    while len(meshes) < i + 1 and meshes:  # frames before her first pose: start from it
        meshes.insert(0, meshes[0])
    meshes = smooth(meshes)
    # Stand her on the floor, in the middle.
    allv = np.stack(meshes)
    floor = np.percentile(allv[:, :, 1].min(1), 5)
    centre = allv[:, :, [0, 2]].mean((0, 1))
    meshes = [m - np.array([centre[0], floor, centre[1]], np.float32) for m in meshes]

    clip_end = start + len(meshes) / fps
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', f'{start:.3f}', '-to', f'{clip_end:.3f}', '-i', video,
                        '-vn', '-c:a', 'aac', '-b:a', '160k', '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov', 'pipe:1'],
                       capture_output=True)
    audio = r.stdout if r.returncode == 0 else b''

    os.makedirs(out_dir, exist_ok=True)
    safe = ''.join('_' if c in '\\/:*?"<>|' else c for c in name)
    path = os.path.join(out_dir, f'{safe}.fancam3d')
    write_file(path, name, fps, meshes, est.faces, audio)
    print('완성:', path)
    return {'path': path, 'frames': len(meshes), 'duration': len(meshes) / fps, 'fixes': follower.fixes,
            'partial': len(meshes) < total}
