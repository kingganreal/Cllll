"""Processing server for the app (직캠 스테이지 → 만들기 → 코랩 GPU로 처리).

Runs in the Colab notebook next to fancam.py; cloudflared gives it a public https address, which
goes into the app together with a random key (every request must carry it). One job at a time.

  GET  /status                         is it up? (GPU name)
  POST /jobs            JSON params    new job -> {id} (with params.url the server downloads the video)
  PUT  /jobs/<id>/part?i=N   bytes     the video, in pieces (a quick tunnel takes 100 MB at most)
  POST /jobs/<id>/start                all pieces are in: queue it
  GET  /jobs/<id>                      {state, done, total, stage, error}
  POST /jobs/<id>/stop                 finish early, keeping what's made (여기까지 저장)
  DELETE /jobs/<id>                    cancel and forget
  GET  /jobs/<id>/live?from=A&to=B     the clip so far while it is made (circle mode), X-Final/X-Size
  GET  /jobs/<id>/info                 the finished clip's size and details
  GET  /jobs/<id>/result               the clip (packed MP4), whole or a Range of it (the app fetches
                                       several ranges at once: a quick tunnel is slow per connection)
"""
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fancam

ROOT = '/content/jobs' if os.path.isdir('/content') else os.path.join(fancam.CACHE, 'jobs')
KEY = os.environ.get('FANCAM_KEY') or secrets.token_urlsafe(9)
jobs = {}
work = queue.Queue()


DRIVE = '/content/drive/MyDrive'
DRIVE_DIR = '직캠 스테이지/코랩'


def to_drive(job):
    """A copy in Google Drive (when it's connected, notebook step 2): the tunnel can be very slow,
    and from Drive the clip comes at full speed (drive.google.com, or Google Drive on the PC)."""
    if not os.path.isdir(DRIVE):
        return
    try:
        src = job['result']['path']
        os.makedirs(os.path.join(DRIVE, DRIVE_DIR), exist_ok=True)
        dst = os.path.join(DRIVE, DRIVE_DIR, os.path.basename(src))
        shutil.copyfile(src, dst + '.part')
        os.replace(dst + '.part', dst)
        job['drive'] = f'{DRIVE_DIR}/{os.path.basename(src)}'
        print('드라이브에도 저장했어요:', dst)
    except Exception as err:
        print('드라이브에 저장하지 못했어요:', err)


def worker():
    while True:
        job = work.get()
        if job['state'] == 'cancelled':
            continue
        job['state'] = 'running'
        try:
            p = job['params']
            if p.get('url'):
                job['state'] = 'downloading'
                try:
                    src = fancam.download_youtube(p['url'], job['dir'])
                except Exception as err:
                    # YouTube often refuses data-centre addresses; the app then sends the file itself.
                    job['state'], job['error'] = 'error', 'download: ' + (getattr(err, 'stderr', None) or str(err))[-300:]
                    continue
                job['state'] = 'running'
            else:
                src = os.path.join(job['dir'], 'source')
                with open(src, 'wb') as out:
                    for i in range(job['parts']):
                        with open(os.path.join(job['dir'], f'part{i}'), 'rb') as f:
                            shutil.copyfileobj(f, out)
                        os.remove(os.path.join(job['dir'], f'part{i}'))

            def progress(done, total, stage=None):
                job.update(done=done, total=total, stage=stage)

            common = dict(name=p.get('name') or '클립', start=float(p.get('start', 0)), end=p.get('end'),
                          max_side=int(p.get('maxSide', 1280)), ref=p.get('faceRef'), pick=p.get('pick'), on_progress=progress,
                          should_stop=lambda: job['stop'], cancelled=lambda: job['state'] == 'cancelled')
            if p.get('shape') == '3d':
                import body3d
                r = body3d.make_3d(src, job['dir'], fps=int(p.get('fps3d', 15)), **common)
                r.update(kind='3d', layout=None, width=None, height=None)
            else:
                r = fancam.make_clip(
                    src, job['dir'], fps=int(p.get('fps', 30)), shape=p.get('shape', 'cutout'),
                    sam_model=p.get('sam') or os.environ.get('FANCAM_SAM', 'auto'), tracker=p.get('tracker') or 'sam',
                    on_output=lambda path: job.update(live=path), **common)
                r.update(kind='clip')
            job['result'] = r
            job['state'] = 'done'
            threading.Thread(target=to_drive, args=(job,), daemon=True).start()
            try:
                os.remove(src)
            except OSError:
                pass
        except fancam.Stopped:
            job['state'] = 'cancelled'
        except Exception as err:
            traceback.print_exc()
            job['state'], job['error'] = 'error', str(err)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Key, Range')
        self.send_header('Access-Control-Expose-Headers', 'X-Clip, X-Final, X-Size')

    def send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.cors()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(n) if n else b''

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def route(self, method):
        if not secrets.compare_digest(self.headers.get('X-Key', ''), KEY):
            self.body()
            return self.send(403, {'error': '키가 맞지 않아요. 코랩에 나온 주소를 다시 붙여 넣어 주세요.'})
        path = self.path.split('?')[0]
        m = re.match(r'^/jobs/([\w-]+)(/[\w]+)?$', path)
        job = jobs.get(m.group(1)) if m else None
        sub = m.group(2) if m else None
        if path == '/status' and method == 'GET':
            import torch
            return self.send(200, {'ok': True, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   'busy': sum(j['state'] in ('queued', 'running') for j in jobs.values())})
        if path == '/jobs' and method == 'POST':
            params = json.loads(self.body() or b'{}')
            jid = uuid.uuid4().hex[:12]
            d = os.path.join(ROOT, jid)
            os.makedirs(d, exist_ok=True)
            jobs[jid] = {'id': jid, 'dir': d, 'params': params, 'state': 'uploading', 'parts': 0, 'stop': False,
                         'done': 0, 'total': 0, 'stage': None, 'error': None, 'result': None, 'created': time.time()}
            return self.send(200, {'id': jid})
        if job is None:
            self.body()
            return self.send(404, {'error': '없는 작업이에요. 코랩이 다시 시작됐을 수 있어요.'})
        if sub == '/part' and method == 'PUT':
            i = int(re.search(r'i=(\d+)', self.path).group(1))
            with open(os.path.join(job['dir'], f'part{i}'), 'wb') as f:
                f.write(self.body())
            job['parts'] = max(job['parts'], i + 1)
            return self.send(200, {'ok': True})
        if sub == '/start' and method == 'POST':
            self.body()
            job['state'] = 'queued'
            work.put(job)
            return self.send(200, {'ok': True})
        if sub is None and method == 'GET':
            r = job['result']
            return self.send(200, {k: job[k] for k in ('state', 'done', 'total', 'stage', 'error')} | {'fixes': r and r['fixes']})
        if sub == '/stop' and method == 'POST':
            self.body()
            job['stop'] = True
            return self.send(200, {'ok': True})
        if sub is None and method == 'DELETE':
            job['state'] = 'cancelled'
            threading.Timer(30, lambda: (jobs.pop(job['id'], None), shutil.rmtree(job['dir'], ignore_errors=True))).start()
            return self.send(200, {'ok': True})
        if sub == '/live' and method == 'GET':
            # The clip while it's being made (it only grows): bytes ?from= up to ?to= (or 16 MB), as
            # far as written. X-Size: written so far (0: nothing yet); X-Final: 1 once it's complete.
            r = job['result']
            path = r['path'] if r else job.get('live')
            if not path or not os.path.exists(path):
                if job['state'] not in ('queued', 'running', 'downloading', 'uploading'):
                    return self.send(409, {'error': '클립이 없어요.'})
                path = None
            q = lambda k, d: int((re.search(k + r'=(\d+)', self.path) or [0, d])[1])
            size = os.path.getsize(path) if path else 0
            a = q('from', 0)
            limit = max(a, min(size, q('to', a + (16 << 20))))
            self.send_response(200)
            self.cors()
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(max(0, limit - a)))
            self.send_header('X-Final', '1' if r else '0')
            self.send_header('X-Size', str(size))
            self.end_headers()
            if limit > a:
                with open(path, 'rb') as f:
                    f.seek(a)
                    left = limit - a
                    while left > 0:
                        chunk = f.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            return
        if sub in ('/result', '/info') and method == 'GET':
            r = job['result']
            if not r:
                return self.send(409, {'error': '아직 안 끝났어요.'})
            size = os.path.getsize(r['path'])
            meta = {k: r.get(k) for k in ('kind', 'layout', 'width', 'height', 'duration', 'fixes', 'partial', 'frames')}
            if sub == '/info':
                return self.send(200, {'size': size, 'meta': meta, 'drive': os.path.isdir(DRIVE) and f'{DRIVE_DIR}/{os.path.basename(r["path"])}'})
            rng = re.match(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
            a = int(rng.group(1)) if rng else 0
            b = min(size - 1, int(rng.group(2))) if rng and rng.group(2) else size - 1
            self.send_response(206 if rng else 200)
            self.cors()
            self.send_header('Content-Type', 'application/octet-stream' if r.get('kind') == '3d' else 'video/mp4')
            self.send_header('Content-Length', str(b - a + 1))
            if rng:
                self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
            self.send_header('X-Clip', json.dumps(meta))
            self.end_headers()
            with open(r['path'], 'rb') as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return
        return self.send(404, {'error': 'not found'})

    def do_GET(self): self.route('GET')
    def do_POST(self): self.route('POST')
    def do_PUT(self): self.route('PUT')
    def do_DELETE(self): self.route('DELETE')


def cloudflared():
    """The cloudflared program (downloaded once, before any tunnel starts)."""
    exe = shutil.which('cloudflared') or os.path.join(fancam.CACHE, 'cloudflared')
    if not os.path.exists(exe):
        fancam.fetch('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', 'cloudflared')
        os.chmod(exe, 0o755)
    return exe


def tunnel(port):
    """A public https address for the port through a Cloudflare quick tunnel."""
    exe = cloudflared()
    proc = subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--url', f'http://localhost:{port}'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        m = re.search(r'https://[\w-]+\.trycloudflare\.com', line)
        if m:
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
            return m.group(0)
    raise RuntimeError('공개 주소를 만들지 못했어요.')


def serve(port=8765, public=True, tunnels=4):
    os.makedirs(ROOT, exist_ok=True)
    # Running this cell again: stop the server and tunnels from the last run first (kept on
    # builtins, so they're found even after the module was re-imported).
    import builtins
    old = getattr(builtins, '_fancam_httpd', None)
    if old is not None:
        try:
            old.shutdown(); old.server_close()
        except Exception:
            pass
    if public and shutil.which('pkill'):
        subprocess.run(['pkill', '-f', 'cloudflared tunnel'], check=False)
    threading.Thread(target=worker, daemon=True).start()
    for k in range(20):  # still taken (another notebook?): the next port
        try:
            httpd = ThreadingHTTPServer(('0.0.0.0', port + k), Handler)
            port += k
            break
        except OSError as err:
            if err.errno != 98 or k == 19:
                raise
            time.sleep(0.5 if k == 0 else 0)
    builtins._fancam_httpd = httpd
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if public:
        # One quick tunnel is slow, so several are opened side by side and the app spreads its
        # pieces over them (the extra ones ride along in the address as &more=).
        cloudflared()
        found = [None] * tunnels

        def one(k):
            try:
                found[k] = tunnel(port)
            except Exception as err:
                print('터널 하나를 못 열었어요:', err)
        ts = [threading.Thread(target=one, args=(k,)) for k in range(tunnels)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(90)
        bases = [b for b in found if b]
        if not bases:
            raise RuntimeError('공개 주소를 만들지 못했어요.')
    else:
        bases = [f'http://localhost:{port}']
    link = f'{bases[0]}/#key={KEY}' + ('&more=' + ','.join(b.split('//')[1] for b in bases[1:]) if len(bases) > 1 else '')
    print('\n앱의 만들기 → "코랩 GPU로 처리"에 이 주소를 붙여 넣으세요:\n\n   ' + link + '\n')
    print('이 칸이 실행 중인 동안 앱에서 쓸 수 있어요. 코랩을 다시 켜면 주소가 바뀌어요.')
    return link



def wait():
    """Keeps the notebook cell running (Colab cuts off a runtime whose cells have all finished,
    even while this server is working) and shows what it's doing. Stop the cell to end it."""
    last = None
    try:
        while True:
            busy = [j for j in jobs.values() if j['state'] in ('uploading', 'queued', 'downloading', 'running')]
            if busy:
                j = busy[0]
                done, total = j.get('done') or 0, j.get('total') or 0
                line = f"{j['params'].get('name') or '클립'}: " + (
                    {'uploading': '영상 받는 중', 'queued': '차례 기다리는 중', 'downloading': '유튜브에서 받는 중'}.get(j['state'])
                    or j.get('stage') or (f'{done}/{total} 프레임 ({done * 100 // max(1, total)}%)' if total else '준비 중'))
            else:
                line = '기다리는 중 (앱에서 추출 시작을 누르세요)'
            if line != last:
                print(time.strftime('%H:%M:%S'), line, flush=True)
                last = line
            time.sleep(20)
    except KeyboardInterrupt:
        print('서버를 멈췄어요. 다시 쓰려면 이 칸을 다시 실행하세요.')
        old = getattr(__import__('builtins'), '_fancam_httpd', None)
        if old is not None:
            old.shutdown(); old.server_close()


if __name__ == '__main__':
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765, public='--local' not in sys.argv)
    while True:
        time.sleep(3600)
