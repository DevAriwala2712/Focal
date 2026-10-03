// Shared shell: header, navigation, data loading and formatting helpers. Every number shown on a page comes from data/site.json.
(function () {
  const NAV = [
    ['how-it-works', 'How it works', 'info'],
    ['data-dates', 'Data & dates', 'calendar_today'],
    ['map-workspace', 'Map workspace', 'map'],
    ['objects', 'Objects', 'category'],
    ['evidence', 'Evidence', 'verified'],
    ['experiments', 'Experiments', 'science'],
    ['demo', 'Demo', 'play_circle'],
    ['exports', 'Exports', 'output'],
  ];

  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const int = (n) => (n == null || Number.isNaN(n) ? '—' : Math.round(n).toLocaleString('en-US'));
  const fix = (n, d = 2) => (n == null || Number.isNaN(n) ? '—' : Number(n).toFixed(d));
  const signed = (n, d = 3) => (n == null ? '—' : (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(d));
  const bytes = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' kB' : n + ' B');
  const chip = (kind, glyph, text) => `<span class="chip ${kind}"><span style="font-size:9px">${glyph}</span>${esc(text)}</span>`;
  const statusChip = (s) => {
    const u = String(s || '').toUpperCase();
    if (u === 'PASS' || u === 'KEEP') return chip('ok', '●', u);
    if (u === 'FAIL' || u === 'DOWNGRADE') return chip('warn', '▲', u);
    return chip('ref', '■', u || 'NOT RUN');
  };

  const ready = fetch('data/site.json').then((r) => {
    if (!r.ok) throw new Error('data/site.json missing – run scripts/build_web_data.py');
    return r.json();
  });

  function shell(site) {
    const page = document.body.dataset.page;
    const c = site.aoi_centre;
    const header = `
<header class="fixed top-0 left-0 right-0 h-14 bg-surface-container border-b border-outline-variant z-50 flex items-center justify-between px-margin select-none">
  <div class="flex items-center gap-space-md min-w-0">
    <div class="flex items-baseline gap-space-xs"><span class="font-headline-sm text-headline-sm text-primary font-bold tracking-tight uppercase">TrustSR</span>
      <span class="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant px-space-xs py-0.5 bg-surface-container-low">evidence run</span></div>
    <div class="h-4 w-px bg-outline-variant mx-space-xs"></div>
    <div class="hidden md:flex items-center gap-space-sm bg-surface-container-low border border-outline-variant px-space-sm py-1">
      <span class="font-label-md text-label-md text-on-surface">Wayanad debris flow · 30 Jul 2024</span>
      <span class="font-label-sm text-label-sm text-on-surface-variant">AOI centre ${c.lat.toFixed(3)}°N, ${c.lon.toFixed(3)}°E</span></div>
    <div class="hidden xl:flex items-center gap-space-xs">
      ${chip('ok', '●', 'Evidence: real')}
      ${chip('warn', '▲', 'Retrospective, not rapid response')}
      ${chip('ref', '■', 'Model: pretrained, not fine-tuned')}
      ${chip('warn', '▲', 'Gate: k=' + site.k.toFixed(1) + ', not calibrated')}
    </div>
  </div>
  <div class="hidden lg:flex items-center gap-space-md">
    <div class="flex items-center gap-space-sm font-label-sm text-label-sm text-on-surface-variant bg-surface-container-low border border-outline-variant px-space-sm py-1">
      <span class="material-symbols-outlined text-[14px]">my_location</span><span>${esc(site.crs)} · UTM 43N</span></div>
    <div class="flex items-center gap-space-xs font-label-sm text-label-sm px-space-sm py-1 bg-surface-container-high border border-outline"><span class="w-1.5 h-1.5 bg-primary"></span><span class="text-on-surface">Read-only</span></div>
  </div>
</header>`;
    const side = `
<aside class="fixed left-0 top-14 bottom-0 w-56 bg-surface-container-low border-r border-outline-variant z-40 flex flex-col justify-between py-space-sm">
  <div class="flex flex-col">
    <div class="px-space-md py-space-xs mb-space-xs border-b border-outline-variant"><span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Navigation index</span></div>
    <nav class="flex flex-col gap-0.5">${NAV.map(([id, label, icon]) => id === page
      ? `<a aria-current="page" href="${id}.html" class="flex items-center gap-space-sm px-space-md py-space-sm bg-surface-container border-l-2 border-primary text-primary font-medium font-body-sm text-body-sm"><span class="material-symbols-outlined text-[18px]">${icon}</span><span>${label}</span></a>`
      : `<a href="${id}.html" class="flex items-center gap-space-sm px-space-md py-space-sm text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface border-l-2 border-transparent font-body-sm text-body-sm"><span class="material-symbols-outlined text-[18px]">${icon}</span><span>${label}</span></a>`).join('')}
    </nav>
  </div>
  <div class="p-space-md border-t border-outline-variant flex flex-col gap-space-xs font-label-sm text-label-sm text-on-surface-variant">
    <div class="flex justify-between"><span>GRID:</span><span class="text-on-surface">10 m → 2.5 m (×4)</span></div>
    <div class="flex justify-between"><span>BANDS:</span><span class="text-on-surface">B02 B03 B04 B08</span></div>
    <div class="flex justify-between"><span>CFG SHA:</span><span class="text-on-surface">${esc(site.config_sha256.slice(0, 8))}…</span></div>
  </div>
</aside>`;
    document.body.insertAdjacentHTML('afterbegin', header + side);
    const main = document.getElementById('app');
    main.classList.add('pl-56', 'pt-14');
  }

  window.TSR = { ready, esc, int, fix, signed, bytes, chip, statusChip, NAV };
  document.addEventListener('DOMContentLoaded', () => {
    ready.then((site) => { window.TSR.site = site; shell(site); document.dispatchEvent(new CustomEvent('tsr-ready', { detail: site })); })
      .catch((e) => { document.body.innerHTML = `<pre class="code" style="margin:24px">${esc(e.message)}\nServe demo/web with: python demo/server.py</pre>`; });
  });
})();
