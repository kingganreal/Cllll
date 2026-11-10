"""직캠 스테이지 for Google Colab: cuts one member out of a fancam on a cloud GPU.

The same idea as the web app (js/processor.js), with the full-size models it can't run in a
browser: SAM 2.1 (base+, all 7 memory frames) follows her, Robust Video Matting (ResNet-50) draws
her edges, and face recognition (InsightFace buffalo_s) keeps the tracker on her. The result is
the app's own clip format (colour | alpha packed in one MP4, "<name>.stage-<layout>.mp4"), so the
app's "파일 가져오기" takes it as is.

SAM 2 reads the frames as it goes (its own video loader keeps every frame in memory, which a long
fancam doesn't fit), and when the face check finds that the tracker has drifted onto someone
else, tracking starts again there from her face.
"""
import json
import math
import os
import shutil
import subprocess
import urllib.request
import zipfile

import cv2
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# bfloat16 halves SAM 2's time on GPUs that have it (Ampere and newer; Colab's T4 doesn't).
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
CACHE = os.path.expanduser('~/.cache/fancam')

SAM_MODELS = {  # name -> (config, checkpoint URL)
    'tiny': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'),
    'small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'),
    'base_plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'),
    'large': ('configs/sam2.1/sam2.1_hiera_l.yaml', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'),
}
FACE_ZIP = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip'


def fetch(url, name=None):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name or url.rsplit('/', 1)[-1])
    if not os.path.exists(path):
        print('받는 중:', url)
        tmp = path + '.part'
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Faces (SCRFD-500M + MobileFaceNet, the same models and maths as js/face.js)

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
SAME, MARGIN, OTHER, OTHER_LEARNED, LEARN, LEARN_MARGIN = 0.42, 0.08, 0.25, 0.35, 0.4, 0.12


class Faces:
    def __init__(self):
        import onnxruntime as ort
        z = fetch(FACE_ZIP)
        d = os.path.join(CACHE, 'buffalo_s')
        if not os.path.exists(os.path.join(d, 'w600k_mbf.onnx')):
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
        find = lambda n: next(os.path.join(r, n) for r, _, fs in os.walk(d) if n in fs)
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.det = ort.InferenceSession(find('det_500m.onnx'), providers=prov)
        self.rec = ort.InferenceSession(find('w600k_mbf.onnx'), providers=prov)

    def __call__(self, bgr, min_size=None):
        """Every face: dicts with box, kps (5 points), size (eye-to-mouth px), emb (unit 512-d)."""
        h, w = bgr.shape[:2]
        D = 640
        s = D / max(w, h)
        img = np.zeros((D, D, 3), np.uint8)
        img[:round(h * s), :round(w * s)] = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        x = ((img[:, :, ::-1].astype(np.float32) - 127.5) / 128).transpose(2, 0, 1)[None]
        o = self.det.run(None, {self.det.get_inputs()[0].name: x})
        found = []
        for si, st in enumerate([8, 16, 32]):
            sc, bx, kp = o[si].reshape(-1), o[si + 3].reshape(-1, 4), o[si + 6].reshape(-1, 10)
            gw = D // st
            for a in np.where(sc >= 0.5)[0]:
                cell = a >> 1
                cx, cy = (cell % gw) * st, (cell // gw) * st
                b = np.array([cx - bx[a, 0] * st, cy - bx[a, 1] * st, cx + bx[a, 2] * st, cy + bx[a, 3] * st]) / s
                k = np.array([[cx + kp[a, 2 * p] * st, cy + kp[a, 2 * p + 1] * st] for p in range(5)]) / s
                found.append(dict(box=b, kps=k, score=float(sc[a])))
        found.sort(key=lambda f: -f['score'])
        kept = []
        for f in found:
            if all(_iou(k['box'], f['box']) < 0.4 for k in kept):
                kept.append(f)
        min_size = min_size or max(w, h) * 0.012
        kept = [f for f in kept if min(f['box'][2] - f['box'][0], f['box'][3] - f['box'][1]) >= min_size]
        if not kept:
            return []
        crops = []
        for f in kept:
            M, _ = cv2.estimateAffinePartial2D(f['kps'].astype(np.float32), TEMPLATE, method=cv2.LMEDS)
            c = cv2.warpAffine(bgr, M, (112, 112), borderValue=0)
            crops.append(((c[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1))
        e = self.rec.run(None, {self.rec.get_inputs()[0].name: np.stack(crops)})[0]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        for f, v in zip(kept, e):
            f['emb'] = v
            f['size'] = float((f['kps'][3][1] + f['kps'][4][1]) / 2 - (f['kps'][0][1] + f['kps'][1][1]) / 2)
        return kept


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def frontal(f):
    le, re, nose = f['kps'][0], f['kps'][1], f['kps'][2]
    eyes = np.linalg.norm(re - le)
    t = float(np.dot(nose - le, re - le) / (eyes * eyes + 1e-6))
    return eyes > 0.3 * (f['box'][2] - f['box'][0]) and 0.25 < t < 0.75


class Matcher:
    """Her face (see js/face.js matcher): photos, plus how she looks in this video once sure."""

    def __init__(self, ref):
        self.ref, self.sum, self.seen, self.video = ref, np.zeros(512, np.float32), 0, None

    def learn(self, emb):
        self.sum += emb
        self.seen += 1
        self.video = self.sum / np.linalg.norm(self.sum)

    def find(self, found, plausible=lambda f: True):
        best, second = None, -1
        for f in found:
            f['photo'] = float(f['emb'] @ self.ref)
            f['sim'] = max(f['photo'], float(f['emb'] @ self.video)) if self.seen >= 2 else f['photo']
            if best is None or f['sim'] > best['sim']:
                if best is not None:
                    second = best['sim']
                best = f
            elif f['sim'] > second:
                second = f['sim']
        her = best if best is not None and best['sim'] >= SAME and best['sim'] - second >= MARGIN else None
        if her is not None and not plausible(her):
            her = None
        if her is not None and her['photo'] >= LEARN and her['sim'] - second >= LEARN_MARGIN:
            self.learn(her['emb'])
        return her

    def other(self, f):
        return f['sim'] < (OTHER_LEARNED if self.seen >= 5 else OTHER)


def reference_from_photos(faces, paths):
    embs = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fs = faces(img, min_size=24)
        if fs:
            embs.append(max(fs, key=lambda f: f['box'][2] - f['box'][0])['emb'])
    if not embs:
        return None
    v = np.mean(embs, 0)
    print(f'사진 {len(embs)}장에서 얼굴을 찾았어요.')
    return v / np.linalg.norm(v)


def reference_from_video(faces, path, samples=200):
    """No photos: in a fancam the member it follows is the face seen most often. Over the whole
    video, not just the part being cut: in a few seconds another member can outnumber her."""
    cap = cv2.VideoCapture(path)
    start, end = 0.0, probe(path)[2]
    embs, sizes = [], []
    for t in np.linspace(start, end, samples, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        for f in faces(fr):
            if frontal(f):
                embs.append(f['emb'])
                sizes.append(f['size'])
    if not embs:
        return None
    E = np.stack(embs)
    clusters = []
    for i in np.argsort(sizes)[::-1]:
        best, bs = None, 0.35
        for c in clusters:
            s = float(c['c'] @ E[i])
            if s > bs:
                best, bs = c, s
        if best is None:
            clusters.append(dict(c=E[i].copy(), m=[i]))
        else:
            best['m'].append(i)
            v = E[best['m']].sum(0)
            best['c'] = v / np.linalg.norm(v)
    big = max(clusters, key=lambda c: len(c['m']))
    print(f'사진이 없어서 영상에서 가장 많이 나온 얼굴로 정했어요 ({len(big["m"])}/{len(embs)}번).')
    return big['c']


# ---------------------------------------------------------------------------
# Video in and out

class Frames:
    """Frames at output times start + (i + 0.5) / fps, at `size` (w, h), read once and cached."""

    def __init__(self, path, start, end, fps, size):
        self.cap = cv2.VideoCapture(path)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start - 1) * 1000)
        self.pos = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.start, self.fps, self.size = start, fps, size
        self.total = max(1, int((end - start) * fps))
        self.cache, self.full, self.last = {}, {}, None

    def original(self, i):
        """The same frame at the video's own size (the circle is cut from this, not the working size)."""
        self.get(i)
        return self.full[i]

    def get(self, i):
        if i in self.cache:
            return self.cache[i]
        want = int((self.start + (i + 0.5) / self.fps) * self.src_fps)
        while self.pos <= want:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.pos += 1
            if self.pos > want:
                self.last = fr
        if self.last is None:
            ok, self.last = self.cap.read()
        img = cv2.resize(self.last, self.size, interpolation=cv2.INTER_AREA)
        self.cache[i], self.full[i] = img, self.last
        return img

    def close(self):
        self.cap.release()
        self.cache.clear()
        self.full.clear()

    def forget_before(self, i):
        for k in [k for k in self.cache if k < i]:
            del self.cache[k]
            self.full.pop(k, None)


def probe(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
    return w, h, n / fps


def drop_strays(a, near_kernel):
    """Pieces of the matte floating away from her (a neighbour's raised arm): keep her body (the
    biggest piece), what's near it, and other big pieces; faint edge pixels stay near her body."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((a >= 0.3).astype(np.uint8), connectivity=4)
    if n <= 2:
        return a
    areas = st[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    near = cv2.dilate((lab == best).astype(np.uint8), near_kernel) > 0
    keep = np.zeros(n, bool)
    keep[best] = True
    keep[np.unique(lab[near])] = True
    keep[1:][areas > areas.max() * 0.25] = True
    keep[0] = False
    drop = np.where(lab > 0, ~keep[lab], ~near)
    a = a.copy()
    a[drop] = 0
    return a


def even(v):
    return max(2, int(round(v / 2)) * 2)


# ---------------------------------------------------------------------------
# Main

class Stopped(Exception):
    pass


def make_clip(video, out_dir, name='설윤', photos=(), start=0.0, end=None, fps=30, max_side=1280,
              shape='cutout', sam_model='auto', chunk=300, face_every=0.25, crop=True,
              ref=None, pick=None, on_progress=None, should_stop=None, cancelled=None, tracker='sam'):
    """Makes the clip; returns {'path', 'layout', 'width', 'height', 'duration', 'fixes'}.

    From the app (colab/server.py): `ref` is her face (the registered photos' average embedding,
    or None to skip face checks), `pick` = {'tap': [x, y], 'box': [x0, y0, x1, y1]} where she was
    picked on the first frame (0..1 of the whole frame); `on_progress(done, total, stage)` reports,
    `should_stop()` ends early keeping what's made (여기까지 저장), `cancelled()` aborts.
    """
    import torchvision

    vw, vh, dur = probe(video)
    end = min(end or dur, dur)
    s = max_side / max(vw, vh)
    W, H = even(vw * s), even(vh * s)  # working size
    frames = Frames(video, start, end, fps, (W, H))
    total = frames.total
    print(f'{vw}x{vh} → {W}x{H}, {end - start:.1f}초, {total}프레임, 장치: {DEVICE}')

    report = on_progress or (lambda done, total, stage=None: None)
    report(0, total, '모델 준비 중')
    faces = Faces()
    if ref is not None:
        ref = np.asarray(ref, np.float32)
        ref = ref / np.linalg.norm(ref)
    elif photos:
        ref = reference_from_photos(faces, photos)
    if ref is None and pick is None:
        ref = reference_from_video(faces, video)
        if ref is None:
            raise RuntimeError('영상에서 얼굴을 찾지 못했어요. 설윤 사진을 넣어 주세요.')
    # Picked in the app without registered photos: follow that pick, no face checks.
    her_face = Matcher(ref) if ref is not None else None

    people = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT').to(DEVICE).eval()

    def persons(bgr):
        with torch.no_grad():
            x = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(DEVICE)
            r = people([x])[0]
        keep = (r['labels'] == 1) & (r['scores'] > 0.6)
        return r['boxes'][keep].cpu().numpy()

    def near_middle(f):
        x = (f['box'][0] + f['box'][2]) / 2 / W
        return 0.2 < x < 0.8

    def look(bgr):
        found = faces(bgr)
        if her_face is None:
            return found, None
        return found, her_face.find(found, lambda f: near_middle(f) or f['sim'] >= 0.55)

    def prompt_from_face(bgr, her, found):
        """Box around her body (from the person detector), her face, and her body below it."""
        fx, fy = (her['box'][0] + her['box'][2]) / 2, (her['box'][1] + her['box'][3]) / 2
        fh = her['box'][3] - her['box'][1]
        pts, labels, box = [[fx, fy]], [1], None
        cands = [b for b in persons(bgr) if b[0] <= fx <= b[2] and b[1] - fh <= fy <= b[1] + (b[3] - b[1]) * 0.4]
        if cands:
            box = min(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        for dy in (2.5, 5, 7.5):
            if fy + dy * fh < H * 0.98:
                pts.append([fx, fy + dy * fh]); labels.append(1)
        for f in found:
            if f is not her:
                pts.append([(f['box'][0] + f['box'][2]) / 2, (f['box'][1] + f['box'][3]) / 2]); labels.append(0)
        return np.array(pts, np.float32), np.array(labels, np.int32), (np.array(box, np.float32) if box is not None else None)

    # Find her at the start (or up to 3 s in, where her face shows) - unless she was picked in the app.
    first = None
    if pick is not None:
        tx, ty = pick['tap'][0] * vw, pick['tap'][1] * vh
        pts, labels = [[tx * s, ty * s]], [1]
        box = None
        if pick.get('box'):
            b = pick['box']
            pad = 0.02
            box = np.array([max(0, b[0] - pad) * W, max(0, b[1] - pad) * H, min(1, b[2] + pad) * W, min(1, b[3] + pad) * H], np.float32)
        first = (0, None, None)
        prompt = ('points', (np.array(pts, np.float32), np.array(labels, np.int32), box))
    for i in range(0, min(total, int(3 * fps) + 1) if pick is None else 0, max(1, int(fps / 2))):
        found, her = look(frames.get(i))
        if her is not None:
            first = (i, her, found)
            break
    if first is None:
        raise RuntimeError('처음 3초 안에서 설윤 얼굴을 찾지 못했어요. START를 얼굴이 보이는 곳으로 옮겨 주세요.')
    skip, her, found = first
    if skip:
        print(f'처음 {skip / fps:.1f}초는 얼굴이 안 보여서 거기서부터 시작해요.')
    if pick is None:
        prompt = ('points', prompt_from_face(frames.get(skip), her, found))

    # Matting model
    # Matting model: only the cut-out needs her edges; the circle just follows SAM's mask.
    rvm = None if shape == 'circle' else torch.hub.load('PeterL1n/RobustVideoMatting', 'resnet50' if DEVICE == 'cuda' else 'mobilenetv3', trust_repo=True).to(DEVICE).eval()
    rec = [None] * 4
    ratio = min(1.0, 512 / max(W, H))

    # The circle can follow her with YOLO boxes instead of SAM 2: much faster, a little less sure
    # when members cross with her face hidden.
    use_yolo = shape == 'circle' and tracker == 'yolo'
    if use_yolo:
        from ultralytics import YOLO
        detector = YOLO('yolo11n.pt')
        detector.to(DEVICE)
    else:
        from sam2.build_sam import build_sam2_video_predictor
        if sam_model == 'auto':
            # base+ where the GPU has bfloat16 (A100, L4); on a T4 (float32 only) small is about twice
            # as fast and follows her just as well.
            sam_model = 'base_plus' if BF16 else 'small'
        cfg, url = SAM_MODELS[sam_model]
        predictor = build_sam2_video_predictor(cfg, fetch(url), device=DEVICE)

    # Output: colour and alpha packed like the app does (side by side for portrait, else stacked).
    if shape == 'circle':
        side = even(min(vw, vh, 1080))  # from the original frames, whatever the working size
        ow, oh = side, side
    else:
        ow, oh = W, H
    layout = 'sbs' if oh >= ow else 'tb'
    PW, PH = (ow * 2, oh) if layout == 'sbs' else (ow, oh * 2)
    os.makedirs(out_dir, exist_ok=True)
    safe = ''.join('_' if c in '\\/:*?"<>|' else c for c in name)
    tmp_out = os.path.join(out_dir, f'.{safe}.tmp.mp4')
    first_t = start + skip / fps
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{PW}x{PH}', '-r', str(fps), '-i', '-',
        '-ss', f'{first_t:.3f}', '-t', f'{end - first_t:.3f}', '-i', video,
        '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', tmp_out,
    ], stdin=subprocess.PIPE)

    side_px = max(W, H)
    k_area = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(side_px * 0.015) + 1,) * 2)
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(side_px * 0.01) + 1,) * 2)
    bounds = [1.0, 1.0, 0.0, 0.0]
    cam = None
    disk = None
    if shape == 'circle':
        disk = np.zeros((oh, ow), np.float32)
        cv2.circle(disk, (ow // 2, oh // 2), ow // 2 - 2, 1.0, -1, lineType=cv2.LINE_AA)
        disk = cv2.GaussianBlur(disk, (5, 5), 0)

    def write(i, core, hidden):
        nonlocal rec, cam
        bgr = frames.get(i)
        if rvm is not None:
            with torch.no_grad():
                src = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
                _, pha, *rec = rvm(src, *rec, downsample_ratio=ratio)
            pha = pha[0, 0].float().cpu().numpy()
        else:
            pha = None
        if hidden or core is None:
            a = np.zeros((H, W), np.float32)
        elif pha is None:
            a = core.astype(np.float32)
        else:
            m = core.astype(np.uint8)
            area = cv2.GaussianBlur(cv2.dilate(m, k_area).astype(np.float32), (0, 0), side_px * 0.004)
            inner = cv2.erode(m, k_inner).astype(np.float32)
            a = np.maximum(pha * area, inner)
            a = np.clip((a - 0.1) / 0.85, 0, 1)
            a = drop_strays(a, k_area)
        if shape == 'circle':
            ys, xs = np.nonzero(a[::2, ::2] > 0.5)
            if len(xs) > 20:
                xs, ys = xs * 2, ys * 2
                want = (xs.mean(), ys.min() + (ys.max() - ys.min()) * 0.42,
                        min(max((xs.max() - xs.min()) * 0.7, (ys.max() - ys.min()) * 0.55, min(W, H) * 0.12), min(W, H) / 2))
                cam = list(want) if cam is None else [cam[0] + (want[0] - cam[0]) * 0.2, cam[1] + (want[1] - cam[1]) * 0.2, cam[2] + (want[2] - cam[2]) * 0.08]
            colour = np.zeros((oh, ow, 3), np.uint8)
            alpha = np.zeros((oh, ow), np.float32)
            if cam is not None:
                # Tracked at the working size, cut from the original frame (a 4K source stays sharp).
                sx = vw / W
                rad = cam[2] * sx
                cx, cy = min(max(cam[0] * sx, rad), vw - rad), min(max(cam[1] * sx, rad), vh - rad)
                M = np.array([[ow / (2 * rad), 0, -(cx - rad) * ow / (2 * rad)], [0, oh / (2 * rad), -(cy - rad) * oh / (2 * rad)]], np.float32)
                colour = cv2.warpAffine(frames.original(i), M, (ow, oh), flags=cv2.INTER_AREA if 2 * rad > ow else cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                alpha = disk * (0.0 if hidden else 1.0)
        else:
            colour, alpha = bgr, a
            ys, xs = np.nonzero(a[::2, ::2] > 0.5)
            if len(xs):
                bounds[0] = min(bounds[0], xs.min() * 2 / W); bounds[1] = min(bounds[1], ys.min() * 2 / H)
                bounds[2] = max(bounds[2], xs.max() * 2 / W); bounds[3] = max(bounds[3], ys.max() * 2 / H)
        a8 = (alpha * 255).astype(np.uint8)
        packed = np.zeros((PH, PW, 3), np.uint8)
        if layout == 'sbs':
            packed[:, :ow] = colour; packed[:, ow:] = a8[..., None]
        else:
            packed[:oh] = colour; packed[oh:] = a8[..., None]
        ff.stdin.write(packed.tobytes())

    if not use_yolo:
        # SAM 2 reads frames from here as it needs them, instead of loading the whole video first (that
        # is what takes the memory: ~12 MB a frame at its 1024 size), and memories it won't look at again
        # are dropped as it goes - so a whole fancam runs in one go. Only a re-anchor starts it over.
        size = predictor.image_size
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        class LazyFrames:
            def __init__(self, first):
                self.first = first

            def __len__(self):
                return total - self.first

            def __getitem__(self, k):
                img = cv2.resize(frames.get(self.first + k), (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                return (torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255) - mean) / std

        one = os.path.join(CACHE, 'first')

        def start_state(first, prompt):
            shutil.rmtree(one, ignore_errors=True)
            os.makedirs(one)
            cv2.imwrite(os.path.join(one, '00000.jpg'), frames.get(first))
            state = predictor.init_state(video_path=one, offload_video_to_cpu=True, offload_state_to_cpu=DEVICE == 'cuda')
            state['images'], state['num_frames'] = LazyFrames(first), total - first
            state['video_height'], state['video_width'] = H, W  # masks come back at the working size
            state['cached_features'] = {}
            kind, p = prompt
            if kind == 'mask':
                predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=p)
            else:
                pts, labels, box = p
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pts, labels=labels, box=box)
            return state

        every = max(1, int(round(fps * face_every)))
        fixes, i0, next_i = 0, skip, skip
        off_face, last_her, doubted, hide = 0, None, False, False
        from tqdm.auto import tqdm
        bar = tqdm(total=total - skip, unit='프레임')
        report(skip, total, None)
        stopped = False
        while next_i < total and not stopped:
            restart = None
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=BF16):
                state = start_state(i0, prompt)
                for k, _, logits in predictor.propagate_in_video(state):
                    i = i0 + k
                    if cancelled and cancelled():
                        ff.stdin.close(); ff.wait()
                        raise Stopped('취소됐어요')
                    if should_stop and should_stop() and i > skip:
                        stopped = True
                        break
                    core = logits[0, 0].float().cpu().numpy() > 0
                    # SAM looks back at the picked frame and the last few only.
                    for o in state['output_dict_per_obj'].values():
                        for old in [t for t in o['non_cond_frame_outputs'] if t < k - 20]:
                            del o['non_cond_frame_outputs'][old]

                    # Face check: if the tracker has left her, start again from her face here.
                    if i % every == 0 and her_face is not None:
                        found, her = look(frames.get(i))
                        in_mask = lambda f: np.mean([core[int(min(H - 1, max(0, y))), int(min(W - 1, max(0, x)))] for x, y in f['kps'][:3]]) >= 0.6
                        strangers = [f for f in found if f is not her and her_face.other(f) and frontal(f) and f['size'] > max(W, H) * 0.024 and in_mask(f)]
                        if her is not None:
                            hide, doubted = False, False
                            last_her = ((her['box'][0] + her['box'][2]) / 2, (her['box'][1] + her['box'][3]) / 2, her['box'][2] - her['box'][0], i)
                            off_face = off_face + 1 if not in_mask(her) else 0
                            apart = [f for f in strangers if math.hypot((f['box'][0] + f['box'][2]) / 2 - last_her[0], (f['box'][1] + f['box'][3]) / 2 - last_her[1]) > 1.5 * last_her[2]]
                            if (off_face >= 2 or apart) and k > 0:
                                off_face = 0
                                restart = (i, ('points', prompt_from_face(frames.get(i), her, found)))
                                fixes += 1
                                break
                        elif strangers:
                            near = lambda f: last_her is not None and i - last_her[3] <= fps and math.hypot((f['box'][0] + f['box'][2]) / 2 - last_her[0], (f['box'][1] + f['box'][3]) / 2 - last_her[1]) < 1.5 * last_her[2]
                            if all(near(f) for f in strangers) and not doubted:
                                doubted = True
                            else:
                                hide = True
                        else:
                            doubted = False

                    write(i, core, hide)
                    next_i = i + 1
                    bar.update(1)
                    report(next_i, total, None)
                    frames.forget_before(i)
                predictor.reset_state(state)
            if restart is None:
                break
            i0, prompt = restart
    else:
        # Circle with the fast tracker (as in js/processor.js trackWithYolo): YOLO finds everyone
        # each frame; her box is the one her face is in (every quarter second), else the one that
        # continues her last box and has her colours; a stranger's face in her box sends the
        # circle back to where her face last was.
        one = None
        every = max(1, int(round(fps * face_every)))
        fixes, next_i = 0, skip
        last_her, doubted = None, False
        from tqdm.auto import tqdm
        bar = tqdm(total=total - skip, unit='프레임')
        report(skip, total, None)

        def box_iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            i_ = ix * iy
            return i_ / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i_ + 1e-9)

        def hist_of(b, bgr):
            x0, x1 = int(b[0] + (b[2] - b[0]) * 0.25), int(b[2] - (b[2] - b[0]) * 0.25)
            y0, y1 = int(b[1] + (b[3] - b[1]) * 0.2), int(b[3] - (b[3] - b[1]) * 0.2)
            patch = bgr[max(0, y0):max(1, y1):3, max(0, x0):max(1, x1):3][:, :, ::-1].reshape(-1, 3) >> 6
            h = np.bincount(patch[:, 0] * 16 + patch[:, 1] * 4 + patch[:, 2], minlength=64).astype(np.float32)
            return h / max(1, h.sum())

        alike = lambda a, b: float(np.sqrt(a * b).sum())
        box = None
        if pick is not None and pick.get('box'):
            b = pick['box']
            box = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
        elif first[1] is not None:  # found by face in this notebook
            box = None
        her_box, her_hist, frozen, vel = box, None, False, [0.0, 0.0]
        tap = (pick['tap'][0] * W, pick['tap'][1] * H) if pick is not None else None
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        for i in range(skip, total):
            if cancelled and cancelled():
                ff.stdin.close(); ff.wait()
                raise Stopped('취소됐어요')
            if should_stop and should_stop() and i > skip:
                break
            bgr = frames.get(i)
            res = detector(bgr, classes=[0], conf=0.3, verbose=False)[0]
            boxes = [list(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()]
            choice = None
            if box is None:
                if tap is not None:
                    inside = [b for b in boxes if b[0] < tap[0] < b[2] and b[1] < tap[1] < b[3]]
                    choice = min(inside, key=area) if inside else None
                elif first[1] is not None:
                    her0 = first[1]
                    fx, fy = (her0['box'][0] + her0['box'][2]) / 2, (her0['box'][1] + her0['box'][3]) / 2
                    inside = [b for b in boxes if b[0] < fx < b[2] and b[1] < fy < b[1] + (b[3] - b[1]) * 0.45]
                    choice = min(inside, key=area) if inside else None
            if her_face is not None and i % every == 0:
                found, her = look(bgr)
                in_box = lambda f, b: b[0] < (f['box'][0] + f['box'][2]) / 2 < b[2] and b[1] < (f['box'][1] + f['box'][3]) / 2 < b[1] + (b[3] - b[1]) * 0.45
                if her is not None:
                    last_her = ((her['box'][0] + her['box'][2]) / 2, (her['box'][1] + her['box'][3]) / 2, her['box'][2] - her['box'][0], i)
                    doubted = False
                    mine = [b for b in boxes if in_box(her, b)]
                    if mine:
                        m = min(mine, key=area)
                        if box is not None and box_iou(m, box) < 0.3:
                            fixes += 1
                        choice, her_box, frozen = m, m, False
                        h = hist_of(m, bgr)
                        her_hist = h if her_hist is None else her_hist * 0.7 + h * 0.3
                elif box is not None:
                    strangers = [f for f in found if her_face.other(f) and frontal(f) and f['size'] > max(W, H) * 0.024 and in_box(f, box)]
                    if strangers:
                        near = lambda f: last_her is not None and i - last_her[3] <= fps and math.hypot((f['box'][0] + f['box'][2]) / 2 - last_her[0], (f['box'][1] + f['box'][3]) / 2 - last_her[1]) < 1.5 * last_her[2]
                        if all(near(f) for f in strangers) and not doubted:
                            doubted = True
                        elif her_box is not None:
                            box, vel, frozen = list(her_box), [0.0, 0.0], True
                            fixes += 1
                    else:
                        doubted = False
            if her_hist is None and (choice or box):
                her_hist = hist_of(choice or box, bgr)
            if choice is None and box is not None and not frozen:
                guess = [box[0] + vel[0], box[1] + vel[1], box[2] + vel[0], box[3] + vel[1]]
                best = 0.25
                for b in boxes:
                    sc = box_iou(b, guess)
                    if sc > best and (her_hist is None or alike(hist_of(b, bgr), her_hist) > 0.6):
                        choice, best = b, sc
            if choice is not None:
                if box is not None:
                    dx = (choice[0] + choice[2] - box[0] - box[2]) / 2; dy = (choice[1] + choice[3] - box[1] - box[3]) / 2
                    vel = [vel[0] * 0.5 + dx * 0.5, vel[1] * 0.5 + dy * 0.5]
                box = [box[k] + (choice[k] - box[k]) * 0.6 for k in range(4)] if box is not None and box_iou(choice, box) > 0.3 else list(choice)
            else:
                vel = [vel[0] * 0.8, vel[1] * 0.8]
            core = np.zeros((H, W), bool)
            if box is not None:
                core[max(0, int(box[1])):min(H, int(box[3]) + 1), max(0, int(box[0])):min(W, int(box[2]) + 1)] = True
            write(i, core if box is not None else None, False)
            next_i = i + 1
            bar.update(1)
            report(next_i, total, None)
            frames.forget_before(i)
    written = next_i - skip
    frames.close()
    bar.close()
    ff.stdin.close()
    ff.wait()
    if one:
        shutil.rmtree(one, ignore_errors=True)
    print(f'얼굴로 {fixes}번 다시 잡았어요.')

    # Crop the empty space around her (both halves alike), like the app's player does.
    final = os.path.join(out_dir, f'{safe}.stage-{layout}.mp4')
    if crop and shape != 'circle' and bounds[2] > bounds[0]:
        pad = 0.03
        x0, y0 = max(0, bounds[0] - pad), max(0, bounds[1] - pad)
        x1, y1 = min(1, bounds[2] + pad), min(1, bounds[3] + pad)
        cx, cy, cw, ch = even(x0 * ow) // 2 * 2, even(y0 * oh) // 2 * 2, even((x1 - x0) * ow), even((y1 - y0) * oh)
        cw, ch = min(cw, ow - cx), min(ch, oh - cy)
        if layout == 'sbs':
            fc = f'[0:v]split[a][b];[a]crop={cw}:{ch}:{cx}:{cy}[c];[b]crop={cw}:{ch}:{ow + cx}:{cy}[d];[c][d]hstack[v]'
        else:
            fc = f'[0:v]split[a][b];[a]crop={cw}:{ch}:{cx}:{cy}[c];[b]crop={cw}:{ch}:{cx}:{oh + cy}[d];[c][d]vstack[v]'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp_out, '-filter_complex', fc, '-map', '[v]', '-map', '0:a?',
                        '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
                        '-movflags', '+faststart', final], check=True)
        os.remove(tmp_out)
    else:
        os.replace(tmp_out, final)
    print('완성:', final)
    cap = cv2.VideoCapture(final)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {'path': final, 'layout': layout, 'width': fw // 2 if layout == 'sbs' else fw,
            'height': fh if layout == 'sbs' else fh // 2, 'duration': written / fps, 'fixes': fixes,
            'partial': written < total - skip}


def download_youtube(url, out_dir):
    """Up to 4K (the circle is cut from the original frames), preferring VP9 over AV1: OpenCV
    decodes it reliably and faster on Colab's two CPU cores."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, '%(title).80s.%(ext)s')
    r = subprocess.run(['yt-dlp', '-f', 'bv*[height<=2160][vcodec^=vp]+ba/bv*[height<=2160]+ba/b', '--merge-output-format', 'mp4', '-o', out,
                        '--print', 'after_move:filepath', url], check=True, capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('out_dir')
    ap.add_argument('--name', default='설윤'); ap.add_argument('--photos', nargs='*', default=[])
    ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float)
    ap.add_argument('--fps', type=int, default=30); ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--shape', default='cutout'); ap.add_argument('--sam', default='base_plus')
    ap.add_argument('--chunk', type=int, default=300); ap.add_argument('--tracker', default='sam')
    a = ap.parse_args()
    make_clip(a.video, a.out_dir, a.name, a.photos, a.start, a.end, a.fps, a.max_side, a.shape, a.sam, a.chunk, tracker=a.tracker)
