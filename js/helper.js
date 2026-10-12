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

// Resolves to true when the helper is running and reachable from this page.
export async function isRunning() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1500);
    const s = await call('/status', { signal: ctrl.signal });
    clearTimeout(timer);
    return !!s.ok;
  } catch {
    return false;
  }
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
