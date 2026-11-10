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

            r = fancam.make_clip(
                src, job['dir'], name=p.get('name') or '클립', start=float(p.get('start', 0)), end=p.get('end'),
                fps=int(p.get('fps', 30)), max_side=int(p.get('maxSide', 1280)), shape=p.get('shape', 'cutout'),
                sam_model=p.get('sam') or os.environ.get('FANCAM_SAM', 'auto'), tracker=p.get('tracker') or 'sam', ref=p.get('faceRef'), pick=p.get('pick'),
                on_progress=progress, should_stop=lambda: job['stop'], cancelled=lambda: job['state'] == 'cancelled')
            job['result'] = r
            job['state'] = 'done'
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
        self.send_header('Access-Control-Expose-Headers', 'X-Clip')

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
        if sub in ('/result', '/info') and method == 'GET':
            r = job['result']
            if not r:
                return self.send(409, {'error': '아직 안 끝났어요.'})
            size = os.path.getsize(r['path'])
            meta = {k: r[k] for k in ('layout', 'width', 'height', 'duration', 'fixes', 'partial')}
            if sub == '/info':
                return self.send(200, {'size': size, 'meta': meta})
            rng = re.match(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
            a = int(rng.group(1)) if rng else 0
            b = min(size - 1, int(rng.group(2))) if rng and rng.group(2) else size - 1
            self.send_response(206 if rng else 200)
            self.cors()
            self.send_header('Content-Type', 'video/mp4')
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


def tunnel(port):
    """A public https address for the port through a Cloudflare quick tunnel."""
    exe = shutil.which('cloudflared') or os.path.join(fancam.CACHE, 'cloudflared')
    if not os.path.exists(exe):
        fancam.fetch('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', 'cloudflared')
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--url', f'http://localhost:{port}'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        m = re.search(r'https://[\w-]+\.trycloudflare\.com', line)
        if m:
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
            return m.group(0)
    raise RuntimeError('공개 주소를 만들지 못했어요.')


def serve(port=8765, public=True):
    os.makedirs(ROOT, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = tunnel(port) if public else f'http://localhost:{port}'
    link = f'{base}/#key={KEY}'
    print('\n앱의 만들기 → "코랩 GPU로 처리"에 이 주소를 붙여 넣으세요:\n\n   ' + link + '\n')
    print('이 칸이 실행 중인 동안 앱에서 쓸 수 있어요. 코랩을 다시 켜면 주소가 바뀌어요.')
    return link


if __name__ == '__main__':
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765, public='--local' not in sys.argv)
    while True:
        time.sleep(3600)
