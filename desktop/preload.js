// The small, fixed set of things the app's own pages may ask the main process to do.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('fancam', {
  // overlay
  getOverlay: id => ipcRenderer.invoke('overlay:get', id),
  setIgnoreMouse: ignore => ipcRenderer.send('overlay:ignore', ignore),
  drag: (on, offX, offY) => ipcRenderer.send('overlay:drag', on, offX, offY),
  zoom: factor => ipcRenderer.send('overlay:zoom', factor),
  menu: () => ipcRenderer.send('overlay:menu'),
  onState: cb => ipcRenderer.on('state', (e, s) => cb(s)),
  // manager
  clips: () => ipcRenderer.invoke('manager:clips'),
  show: id => ipcRenderer.invoke('manager:show', id),
  hideClip: id => ipcRenderer.invoke('manager:hide-clip', id),
  remove: id => ipcRenderer.invoke('manager:delete', id),
  settings: () => ipcRenderer.invoke('manager:settings'),
  setAutoStart: on => ipcRenderer.invoke('manager:auto-start', on),
  setHidden: h => ipcRenderer.invoke('manager:set-hidden', h),
  openWebApp: () => ipcRenderer.invoke('manager:open-web'),
  onClipsChanged: cb => ipcRenderer.on('clips-changed', cb),
});
