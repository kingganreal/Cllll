// 직캠 스테이지 for Windows: puts clips on the desktop as transparent, always-on-top characters,
// and runs the local server the web app uses for YouTube downloads and for sending clips here.
const { app, BrowserWindow, Tray, Menu, ipcMain, screen, shell, nativeImage, dialog } = require('electron');
const path = require('path');
const { pathToFileURL } = require('url');
const store = require('./store');
const server = require('./server');
const { WEB_APP_URL } = require('./config');

if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}
app.setAppUserModelId('io.github.kingganreal.fancamstage');
// If the graphics driver is blocked, fall back to software WebGL rather than showing nothing.
app.commandLine.appendSwitch('enable-unsafe-swiftshader');

const SIZES = { 작게: 320, 보통: 480, 크게: 720, '아주 크게': 960 };
const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5];
const overlays = new Map(); // overlay id -> { win, s }
let manager = null, tray = null, hidden = false, quitting = false;

const iconPath = () => path.join(__dirname, 'assets', 'icon.png');
const state = () => store.loadState();

// ---------------------------------------------------------------------------
// Characters on the desktop

function clipAspect(clip) {
  const w = (clip.width || 9) * (clip.crop?.w || 1), h = (clip.height || 16) * (clip.crop?.h || 1);
  return w / h;
}

function boundsFor(s, clip) {
  const h = Math.round(s.height);
  const w = Math.max(40, Math.round(h * clipAspect(clip)));
  return { x: Math.round(s.x), y: Math.round(s.y), width: w, height: h };
}

function openOverlay(s) {
  const clip = store.getClip(s.clipId);
  if (!clip) return null;
  const win = new BrowserWindow({
    ...boundsFor(s, clip),
    transparent: true, frame: false, resizable: false, hasShadow: false, skipTaskbar: true,
    alwaysOnTop: s.onTop !== false, backgroundColor: '#00000000', show: !hidden, title: clip.name,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      autoplayPolicy: 'no-user-gesture-required',
      backgroundThrottling: false,
    },
  });
  if (s.onTop !== false) win.setAlwaysOnTop(true, 'floating');
  win.setOpacity(s.opacity || 1);
  // Start click-through; the page turns clicks back on while the cursor is over her.
  win.setIgnoreMouseEvents(true, { forward: true });
  win.loadFile(path.join(__dirname, 'renderer', 'overlay.html'), { query: { id: s.id } });
  win.on('closed', () => overlays.delete(s.id));
  overlays.set(s.id, { win, s });
  return win;
}

function addOverlay(clipId) {
  const clip = store.getClip(clipId);
  if (!clip) return;
  const area = screen.getPrimaryDisplay().workArea;
  const height = SIZES.보통;
  const width = Math.round(height * clipAspect(clip));
  const n = overlays.size;
  const s = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), clipId,
    x: area.x + area.width - width - 40 - n * 60, y: area.y + area.height - height - 10,
    height, flip: false, speed: 1, muted: true, onTop: true, opacity: 1,
  };
  state().overlays.push(s);
  store.saveState();
  if (hidden) setHidden(false);
  openOverlay(s);
}

function removeOverlay(id) {
  const o = overlays.get(id);
  state().overlays = state().overlays.filter(s => s.id !== id);
  store.saveState();
  o?.win.close();
}

function update(id, patch) {
  const o = overlays.get(id);
  if (!o) return;
  Object.assign(o.s, patch);
  store.saveState();
  if ('height' in patch) {
    const clip = store.getClip(o.s.clipId);
    // Grow and shrink around the feet, so she stays standing in the same place.
    const old = o.win.getBounds();
    const b = boundsFor({ ...o.s, x: 0, y: 0 }, clip);
    b.x = Math.round(old.x + old.width / 2 - b.width / 2);
    b.y = old.y + old.height - b.height;
    o.s.x = b.x; o.s.y = b.y;
    o.win.setBounds(b);
  }
  if ('onTop' in patch) o.win.setAlwaysOnTop(!!patch.onTop, 'floating');
  if ('opacity' in patch) o.win.setOpacity(patch.opacity);
  o.win.webContents.send('state', o.s);
}

function setHidden(h) {
  hidden = h;
  for (const { win } of overlays.values()) h ? win.hide() : win.showInactive();
  refreshTray();
}

const fromSender = e => [...overlays.values()].find(o => o.win.webContents === e.sender);

ipcMain.handle('overlay:get', (e, id) => {
  const o = overlays.get(id);
  if (!o) return null;
  const clip = store.getClip(o.s.clipId);
  return { s: o.s, clip: { ...clip, src: pathToFileURL(store.clipPath(clip.id, 'mp4')).href } };
});

ipcMain.on('overlay:ignore', (e, ignore) => {
  const o = fromSender(e);
  if (o) o.win.setIgnoreMouseEvents(ignore, { forward: true });
});

// Dragging: follow the cursor until the page says the button went up.
let drag = null;
ipcMain.on('overlay:drag', (e, on, offX, offY) => {
  const o = fromSender(e);
  if (!o) return;
  clearInterval(drag?.timer);
  drag = null;
  if (!on) {
    const b = o.win.getBounds();
    update(o.s.id, { x: b.x, y: b.y });
    return;
  }
  drag = {
    timer: setInterval(() => {
      if (o.win.isDestroyed()) { clearInterval(drag?.timer); return; }
      const c = screen.getCursorScreenPoint();
      o.win.setPosition(Math.round(c.x - offX), Math.round(c.y - offY));
    }, 16),
  };
});

ipcMain.on('overlay:zoom', (e, factor) => {
  const o = fromSender(e);
  if (o) update(o.s.id, { height: Math.min(2000, Math.max(120, Math.round(o.s.height * factor))) });
});

ipcMain.on('overlay:menu', e => {
  const o = fromSender(e);
  if (!o) return;
  const s = o.s, id = s.id;
  Menu.buildFromTemplate([
    { label: '좌우 반전', type: 'checkbox', checked: !!s.flip, click: () => update(id, { flip: !s.flip }) },
    { label: '소리', type: 'checkbox', checked: !s.muted, click: () => update(id, { muted: !s.muted }) },
    { label: '속도', submenu: SPEEDS.map(v => ({ label: `${v}×`, type: 'radio', checked: s.speed === v, click: () => update(id, { speed: v }) })) },
    { label: '크기', submenu: Object.entries(SIZES).map(([k, v]) => ({ label: k, type: 'radio', checked: s.height === v, click: () => update(id, { height: v }) })) },
    { label: '투명도', submenu: [1, 0.8, 0.6, 0.4].map(v => ({ label: `${Math.round(v * 100)}%`, type: 'radio', checked: (s.opacity || 1) === v, click: () => update(id, { opacity: v }) })) },
    { label: '항상 위에', type: 'checkbox', checked: s.onTop !== false, click: () => update(id, { onTop: s.onTop === false }) },
    { type: 'separator' },
    { label: '하나 더 띄우기', click: () => addOverlay(s.clipId) },
    { label: '이 캐릭터 내리기', click: () => removeOverlay(id) },
    { type: 'separator' },
    { label: '직캠 스테이지 열기', click: showManager },
  ]).popup({ window: o.win });
});

// ---------------------------------------------------------------------------
// Manager window (the list of clips on this PC)

function showManager() {
  if (manager && !manager.isDestroyed()) { manager.show(); manager.focus(); return; }
  manager = new BrowserWindow({
    width: 880, height: 680, minWidth: 420, minHeight: 400, title: '직캠 스테이지', icon: iconPath(),
    backgroundColor: '#110f18', autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  manager.loadFile(path.join(__dirname, 'renderer', 'manager.html'));
  manager.on('close', e => {
    if (quitting) return;
    e.preventDefault();
    manager.hide();
    if (!state().trayHintShown) {
      state().trayHintShown = true;
      store.saveState();
      tray?.displayBalloon?.({ title: '직캠 스테이지', content: '창을 닫아도 작업 표시줄 오른쪽 트레이에서 계속 실행돼요.', iconType: 'info' });
    }
  });
}

function clipsChanged() {
  if (manager && !manager.isDestroyed()) manager.webContents.send('clips-changed');
}

ipcMain.handle('manager:clips', () => store.listClips().map(c => ({
  ...c,
  thumb: require('fs').existsSync(store.clipPath(c.id, 'png')) ? pathToFileURL(store.clipPath(c.id, 'png')).href : null,
  showing: [...overlays.values()].filter(o => o.s.clipId === c.id).length,
})));
ipcMain.handle('manager:show', (e, id) => addOverlay(id));
ipcMain.handle('manager:hide-clip', (e, id) => {
  for (const o of [...overlays.values()]) if (o.s.clipId === id) removeOverlay(o.s.id);
});
ipcMain.handle('manager:delete', (e, id) => {
  for (const o of [...overlays.values()]) if (o.s.clipId === id) removeOverlay(o.s.id);
  store.deleteClip(id);
  clipsChanged();
});
ipcMain.handle('manager:settings', () => ({
  autoStart: app.getLoginItemSettings().openAtLogin, hidden, version: app.getVersion(), webAppUrl: WEB_APP_URL,
}));
ipcMain.handle('manager:auto-start', (e, on) => { app.setLoginItemSettings({ openAtLogin: !!on, args: ['--hidden'] }); });
ipcMain.handle('manager:set-hidden', (e, h) => setHidden(!!h));
ipcMain.handle('manager:open-web', () => shell.openExternal(WEB_APP_URL));

// ---------------------------------------------------------------------------
// Tray

function refreshTray() {
  if (!tray) return;
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: '직캠 스테이지 열기', click: showManager },
    { label: hidden ? '캐릭터 모두 보이기' : '캐릭터 모두 숨기기', click: () => setHidden(!hidden) },
    { label: '웹앱 열기 (클립 만들기)', click: () => shell.openExternal(WEB_APP_URL) },
    { type: 'separator' },
    { label: '끝내기', click: () => { quitting = true; app.quit(); } },
  ]));
}

// ---------------------------------------------------------------------------

app.on('second-instance', showManager);
app.on('before-quit', () => { quitting = true; });
app.on('window-all-closed', () => { /* keep running in the tray */ });

app.whenReady().then(async () => {
  store.ensureDirs();
  tray = new Tray(nativeImage.createFromPath(iconPath()).resize({ width: 16, height: 16 }));
  tray.setToolTip('직캠 스테이지');
  tray.on('click', showManager);
  refreshTray();

  try {
    await server.start({ appVersion: app.getVersion(), onClipsChanged: clipsChanged, onShow: addOverlay });
  } catch (err) {
    dialog.showErrorBox('직캠 스테이지', err.code === 'EADDRINUSE'
      ? '8790 포트를 다른 프로그램이 쓰고 있어요. 예전 PC 도우미(검은 창)가 켜져 있다면 닫고 다시 실행해 주세요.'
      : '로컬 서버를 시작하지 못했어요: ' + err.message);
  }

  // Bring back the characters from last time, dropping ones whose clip is gone.
  state().overlays = state().overlays.filter(s => store.getClip(s.clipId));
  store.saveState();
  for (const s of state().overlays) openOverlay(s);

  if (!process.argv.includes('--hidden')) showManager();
});
