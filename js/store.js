// IndexedDB storage for extracted clips and stage settings.
const DB_NAME = 'fancam-stage';
const DB_VERSION = 1;

let dbPromise;
function db() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains('clips')) d.createObjectStore('clips', { keyPath: 'id' });
      if (!d.objectStoreNames.contains('kv')) d.createObjectStore('kv');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => { dbPromise = null; reject(req.error); };
  });
  return dbPromise;
}

async function run(storeName, mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(storeName, mode);
    const result = fn(tx.objectStore(storeName));
    tx.oncomplete = () => resolve(result?.result ?? result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export const clips = {
  all: async () => {
    const list = await run('clips', 'readonly', s => s.getAll());
    return list.sort((a, b) => b.created - a.created);
  },
  get: id => run('clips', 'readonly', s => s.get(id)),
  put: clip => run('clips', 'readwrite', s => s.put(clip)),
  delete: id => run('clips', 'readwrite', s => s.delete(id)),
};

export const kv = {
  get: key => run('kv', 'readonly', s => s.get(key)),
  set: (key, value) => run('kv', 'readwrite', s => s.put(value, key)),
  delete: key => run('kv', 'readwrite', s => s.delete(key)),
};

// Ask the browser not to evict our data under storage pressure (best effort).
export async function requestPersistence() {
  try { if (navigator.storage?.persist) await navigator.storage.persist(); } catch {}
}
