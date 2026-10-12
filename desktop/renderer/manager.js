const $ = s => document.querySelector(s);
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

async function render() {
  const list = await window.fancam.clips();
  const grid = $('#grid');
  grid.replaceChildren();
  $('#empty').hidden = list.length > 0;
  for (const c of list) {
    const card = document.createElement('article');
    card.className = 'card';
    card.innerHTML = `<div class="thumb"></div><div class="body"><div class="name"></div>
      <div class="meta"></div><div class="actions">
      <button class="btn primary small" data-a="show" type="button">띄우기</button>
      ${c.showing ? '<button class="btn small" data-a="hide" type="button">내리기</button>' : ''}
      <button class="btn danger small" data-a="delete" type="button">삭제</button></div></div>`;
    if (c.thumb) {
      const img = document.createElement('img');
      img.src = c.thumb;
      img.alt = '';
      card.querySelector('.thumb').append(img);
    }
    card.querySelector('.name').textContent = c.name;
    card.querySelector('.meta').textContent = `${fmt(c.duration || 0)}${c.showing ? ` · 바탕화면에 ${c.showing}개` : ''}`;
    card.addEventListener('click', async e => {
      const b = e.target.closest('[data-a]');
      if (!b) return;
      if (b.dataset.a === 'show') await window.fancam.show(c.id);
      if (b.dataset.a === 'hide') await window.fancam.hideClip(c.id);
      if (b.dataset.a === 'delete') {
        if (!b.dataset.confirm) {
          b.dataset.confirm = '1';
          b.textContent = '한 번 더 누르면 삭제';
          setTimeout(() => { delete b.dataset.confirm; b.textContent = '삭제'; }, 3000);
          return;
        }
        await window.fancam.remove(c.id);
      }
      render();
    });
    grid.append(card);
  }
}

async function renderSettings() {
  const s = await window.fancam.settings();
  $('#auto-start').checked = s.autoStart;
  $('#toggle-hidden').textContent = s.hidden ? '캐릭터 모두 보이기' : '캐릭터 모두 숨기기';
  $('#version').textContent = `버전 ${s.version} · 이 PC에서만 쓰는 클립 보관함이에요 (웹앱에서 지워도 여기는 남아요)`;
}

$('#open-web').onclick = () => window.fancam.openWebApp();
$('#auto-start').onchange = e => window.fancam.setAutoStart(e.target.checked);
$('#toggle-hidden').onclick = async () => {
  const s = await window.fancam.settings();
  await window.fancam.setHidden(!s.hidden);
  renderSettings();
};
window.fancam.onClipsChanged(render);
addEventListener('focus', () => { render(); renderSettings(); });
render();
renderSettings();
