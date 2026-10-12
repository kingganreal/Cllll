// Talks to the Windows PC helper (helper/fancam-helper.ps1), which downloads YouTube videos
// in original quality with yt-dlp and serves them on http://localhost:8790.
const BASE = 'http://localhost:8790';

export const isWindowsDesktop = /Windows NT/.test(navigator.userAgent) && !/Mobile/.test(navigator.userAgent);

async function call(path, opts = {}) {
  const res = await fetch(BASE + path, { cache: 'no-store', ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `도우미 오류 (${res.status})`);
  return body;
}

export const DESKTOP_APP_URL = 'https://github.com/kingganreal/Cllll/releases/latest';

// The helper's status ({ ok, desktop, ... }) when it's running and reachable, else null.
export async function status() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1500);
    const s = await call('/status', { signal: ctrl.signal });
    clearTimeout(timer);
    return s.ok ? s : null;
  } catch {
    return null;
  }
}
export const isRunning = async () => !!(await status());

// Is the Windows app (not just the old helper script) running? Cached briefly.
let desktopCheck = null, desktopCheckedAt = 0;
export function desktopRunning() {
  if (!isWindowsDesktop) return Promise.resolve(false);
  if (!desktopCheck || Date.now() - desktopCheckedAt > 15000) {
    desktopCheckedAt = Date.now();
    desktopCheck = status().then(s => !!s?.desktop);
  }
  return desktopCheck;
}

function put(path, body, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('PUT', BASE + path);
    if (onProgress) x.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => (x.status >= 200 && x.status < 300 ? resolve() : reject(new Error(`윈도우 앱 오류 (${x.status})`)));
    x.onerror = () => reject(new Error('윈도우 앱에 연결할 수 없어요. 앱이 켜져 있는지 확인해 주세요.'));
    x.send(body);
  });
}

// Sends a clip to the Windows app and puts it on the desktop.
export async function sendToDesktop(clip, onProgress) {
  const meta = {
    name: clip.name, layout: clip.layout, crop: clip.crop, width: clip.width, height: clip.height,
    duration: clip.duration, hasAudio: clip.hasAudio,
  };
  await put(`/clips/${encodeURIComponent(clip.id)}?meta=${encodeURIComponent(JSON.stringify(meta))}`, clip.blob, onProgress);
  if (clip.thumbnail) await put(`/clips/${encodeURIComponent(clip.id)}/thumb`, clip.thumbnail);
  await call(`/clips/${encodeURIComponent(clip.id)}/show`, { method: 'POST', body: '{}' });
}

export const startDownload = url => call('/download', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }),
}).then(r => r.id);

export const jobState = id => call('/job?id=' + encodeURIComponent(id));

export async function fetchFile(id, title, ext, onProgress) {
  const res = await fetch(`${BASE}/file?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('도우미에서 파일을 가져오지 못했어요.');
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    if (total) onProgress?.(got / total);
  }
  const type = ext === 'webm' ? 'video/webm' : 'video/mp4';
  const safe = (title || 'youtube').replace(/[\\/:*?"<>|]/g, '_').slice(0, 80);
  return new File(chunks, `${safe}.${ext || 'mp4'}`, { type });
}

export const finish = id => call('/done?id=' + encodeURIComponent(id), { method: 'POST', body: '{}' }).catch(() => {});

// A one-file launcher: fetches the latest helper script from this site, then runs it.
export function launcherFile() {
  const site = new URL('.', location.href).href;
  const script = new URL('helper/fancam-helper.ps1', site).href;
  const origin = location.origin;
  const lines = [
    '@echo off',
    'chcp 65001 >nul',
    'title 직캠 스테이지 도우미',
    'set "D=%LOCALAPPDATA%\\FancamStage"',
    'if not exist "%D%" mkdir "%D%"',
    `powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol='Tls12'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '${script}' -OutFile ($env:LOCALAPPDATA + '\\FancamStage\\helper.ps1') } catch { Write-Host '새 버전을 확인하지 못해서 저장된 버전으로 실행해요.' }"`,
    'if not exist "%D%\\helper.ps1" (',
    '  echo 도우미를 받지 못했어요. 인터넷 연결을 확인해 주세요.',
    '  pause',
    '  exit /b 1',
    ')',
    `powershell -NoProfile -ExecutionPolicy Bypass -File "%D%\\helper.ps1" -Origin "${origin}"`,
    'pause',
  ];
  // .bat files need Windows line endings.
  return new Blob([lines.join('\r\n') + '\r\n'], { type: 'application/octet-stream' });
}
