// Pan/zoom raster viewer with before/after swipe, 10 m / 2.5 m switch and a live k-gate over the real Wayanad rasters.
// Rasters come from scripts/build_web_data.py (int16 d and sigma x1e4, uint8 parent, uint8 stored class map at k=2.0).
(function () {
  const CLS = { NO_CHANGE: 0, OBSERVED: 1, INFERRED: 2, UNSUPPORTED: 3, NO_DATA: 255 };
  const COLORS = { 1: [184, 60, 30, 215], 2: [232, 160, 32, 225], 3: [91, 110, 125, 170], 255: [110, 110, 110, 150] };
  const BANDS = ['B04', 'B03', 'B02', 'B08']; // order stored in bands10.u16

  class Viewer {
    constructor(el, site, opts = {}) {
      this.el = el; this.site = site; this.opts = opts;
      [this.H, this.W] = site.crop.shape_2p5m; // rows, cols
      this.h10 = this.H / 4; this.w10 = this.W / 4;
      this.view = { s: 1, tx: 0, ty: 0 };
      this.k = site.k; this.visible = new Set(opts.visible || [1, 2, 255]);
      this.split = opts.split ?? 0.5; this.swipe = opts.swipe !== false;
      this.after = opts.after || '2p5m'; this.listeners = {}; this.raw = {}; this.counts = null;
      this._build();
    }
    on(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
    emit(ev, a) { (this.listeners[ev] || []).forEach((f) => f(a)); }

    _build() {
      const { W, H } = this, el = this.el;
      el.classList.add('overflow-hidden', 'select-none'); if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
      el.style.background = '#dcdad3'; el.style.touchAction = 'none'; el.style.cursor = 'grab';
      const stage = (id) => `<div class="stage absolute left-0 top-0" data-s="${id}" style="width:${W}px;height:${H}px;transform-origin:0 0"></div>`;
      el.innerHTML = `<div class="wrap absolute inset-0" data-w="A">${stage('A')}</div><div class="wrap absolute inset-0" data-w="B">${stage('B')}</div>
        <div class="divider absolute top-0 bottom-0" style="width:2px;background:#1c201a;display:${this.swipe ? 'block' : 'none'}"><div class="absolute" style="left:-13px;top:50%;width:28px;height:36px;margin-top:-18px;background:#fcf9f2;border:1.5px solid #1c201a;cursor:ew-resize;display:flex;align-items:center;justify-content:center;font:11px 'JetBrains Mono'">◄►</div></div>
        <div class="tag-l absolute top-2 left-2 chip ok" style="background:#fcf9f2"></div><div class="tag-r absolute top-2 right-2 chip warn" style="background:#fcf9f2"></div>`;
      this.stA = el.querySelector('[data-s=A]'); this.stB = el.querySelector('[data-s=B]');
      this.wrapB = el.querySelector('[data-w=B]'); this.div = el.querySelector('.divider');
      const img = (src) => { const i = document.createElement('img'); i.className = 'pix absolute left-0 top-0'; i.style.cssText = `width:${W}px;height:${H}px;max-width:none`; i.draggable = false; i.src = src; return i; };
      this.imgs = { before: img('data/layers/pre_10m.png'), a10: img('data/layers/post_10m.png'), a25: img('data/layers/post_sr_2p5m.png') };
      this.stA.appendChild(this.imgs.before); this.stB.append(this.imgs.a10, this.imgs.a25);
      this.canvas = document.createElement('canvas'); this.canvas.width = W; this.canvas.height = H;
      this.canvas.className = 'pix absolute left-0 top-0'; this.canvas.style.cssText = `width:${W}px;height:${H}px;pointer-events:none;opacity:${this.opts.opacity ?? 1}`;
      this.stB.appendChild(this.canvas);
      this.markers = document.createElement('div'); this.markers.className = 'absolute left-0 top-0'; this.markers.style.cssText = `width:${W}px;height:${H}px;pointer-events:none`;
      this.stB.appendChild(this.markers);
      this.sel = document.createElement('div'); this.sel.style.cssText = 'position:absolute;border:1.5px solid #1c201a;outline:1px solid #fff;display:none;pointer-events:none'; this.stB.appendChild(this.sel);
      this.setAfter(this.after); this._bind();
      new ResizeObserver(() => { if (!this._fitted && el.clientWidth) { this._fit(); this._fitted = true; } this._layout(); }).observe(el);
      this._fit(); this._layout();
      const tl = this.site.pre_dates[this.site.pre_dates.length - 1];
      this.tagL = el.querySelector('.tag-l'); this.tagR = el.querySelector('.tag-r');
      this.tagL.textContent = `BEFORE · 10 m · ${tl}`;
      this._tagR();
      this._addMarkers();
      this.ensure(['class_k2.u8']).then(() => this.setK(this.k));
    }
    _tagR() { this.tagR.textContent = `AFTER · ${this.after === '2p5m' ? '2.5 m model reconstruction' : '10 m observation'} · ${this.site.post_date}`; this.tagR.style.display = this.swipe ? '' : 'none'; if (!this.swipe) { this.tagL.style.display = 'none'; } }
    _addMarkers() {
      const pts = this.site.plausibility_points; this.markers.innerHTML = '';
      Object.entries(pts).forEach(([name, p]) => {
        const m = document.createElement('div'); m.dataset.name = name;
        m.style.cssText = `position:absolute;left:${p.px[0]}px;top:${p.px[1]}px;width:0;height:0`;
        m.innerHTML = '<div class="mk" style="position:absolute;left:-7px;top:-7px;width:14px;height:14px;border:1.5px solid #1c201a;background:rgba(252,249,242,.75);transform-origin:7px 7px"></div><div class="mk" style="position:absolute;left:-7px;top:-7px;width:14px;height:14px;font:700 12px/14px JetBrains Mono;text-align:center;color:#1c201a;transform-origin:7px 7px">+</div>';
        this.markers.appendChild(m);
      });
      this._scaleMarkers();
    }
    _scaleMarkers() { const inv = 1 / this.view.s; this.markers.querySelectorAll('.mk').forEach((n) => n.style.transform = `scale(${inv})`); this.sel.style.borderWidth = (1.5 * inv) + 'px'; }
    setAfter(mode) {
      this.after = mode; this.imgs.a10.style.display = mode === '10m' ? '' : 'none'; this.imgs.a25.style.display = mode === '2p5m' ? '' : 'none';
      if (this.tagR) this._tagR();
    }
    setSwipe(on) { this.swipe = on; this.div.style.display = on ? 'block' : 'none'; this.tagL.style.display = this.tagR.style.display = on ? '' : 'none'; this._layout(); }
    setSplit(f) { this.split = Math.min(1, Math.max(0, f)); this._layout(); }
    setOpacity(o) { this.canvas.style.opacity = o; }
    setVisible(set) { this.visible = new Set(set); this._paint(); }
    _fit() { const r = this.el.getBoundingClientRect(); if (!r.width || !r.height) return; const s = Math.min(r.width / this.W, r.height / this.H); this.view = { s, tx: (r.width - this.W * s) / 2, ty: (r.height - this.H * s) / 2 }; this.minS = s * 0.9; }
    _layout() {
      const r = this.el.getBoundingClientRect(), v = this.view, t = `translate(${v.tx}px,${v.ty}px) scale(${v.s})`;
      this.stA.style.transform = this.stB.style.transform = t;
      const x = this.swipe ? this.split * r.width : 0;
      this.wrapB.style.clipPath = this.swipe ? `inset(0 0 0 ${x}px)` : 'none';
      this.div.style.left = x + 'px';
      this._scaleMarkers();
      this.emit('view', v);
    }
    flyTo(col, row, s) { const r = this.el.getBoundingClientRect(); s = s || this.view.s; this.view = { s, tx: r.width / 2 - col * s, ty: r.height / 2 - row * s }; this._layout(); }
    zoomBy(f, cx, cy) {
      const r = this.el.getBoundingClientRect(); cx ??= r.width / 2; cy ??= r.height / 2; const v = this.view;
      const ns = Math.min(40, Math.max(this.minS, v.s * f)); const k = ns / v.s;
      this.view = { s: ns, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k }; this._layout();
    }
    reset() { this._fit(); this._layout(); }
    _bind() {
      const el = this.el; let drag = null, moved = 0;
      el.addEventListener('wheel', (e) => { e.preventDefault(); const r = el.getBoundingClientRect(); this.zoomBy(e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
      el.addEventListener('pointerdown', (e) => {
        const r = el.getBoundingClientRect(); moved = 0;
        if (this.swipe && e.target.closest('.divider')) { drag = { mode: 'split' }; } else { drag = { mode: 'pan', x: e.clientX, y: e.clientY, tx: this.view.tx, ty: this.view.ty }; el.style.cursor = 'grabbing'; }
        el.setPointerCapture(e.pointerId);
      });
      el.addEventListener('pointermove', (e) => {
        const r = el.getBoundingClientRect(); this._hover(e.clientX - r.left, e.clientY - r.top);
        if (!drag) return;
        if (drag.mode === 'split') { this.setSplit((e.clientX - r.left) / r.width); return; }
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; moved = Math.max(moved, Math.abs(dx) + Math.abs(dy));
        this.view.tx = drag.tx + dx; this.view.ty = drag.ty + dy; this._layout();
      });
      el.addEventListener('pointerup', (e) => {
        const wasPan = drag && drag.mode === 'pan'; drag = null; el.style.cursor = 'grab';
        if (wasPan && moved < 4) { const r = el.getBoundingClientRect(); const p = this.toPx(e.clientX - r.left, e.clientY - r.top); if (p) { this.select(p.col, p.row); } }
      });
    }
    toPx(x, y) { const v = this.view; const col = Math.floor((x - v.tx) / v.s), row = Math.floor((y - v.ty) / v.s); return col >= 0 && row >= 0 && col < this.W && row < this.H ? { col, row } : null; }
    _hover(x, y) { const p = this.toPx(x, y); this.emit('hover', p ? { ...p, ...this.lonlat(p.col, p.row) } : null); }
    select(col, row) { this.sel.style.cssText += `;display:block;left:${col}px;top:${row}px;width:1px;height:1px;box-sizing:content-box`; this._scaleMarkers(); this.emit('pick', { col, row }); }
    lonlat(col, row) {
      const c = this.site.crop.corners_lonlat, u = (col + 0.5) / this.W, v = (row + 0.5) / this.H;
      const mix = (i) => (1 - v) * ((1 - u) * c.nw[i] + u * c.ne[i]) + v * ((1 - u) * c.sw[i] + u * c.se[i]);
      return { lon: mix(0), lat: mix(1) };
    }

    // ---- rasters ----
    ensure(names) {
      this.pend ||= {};
      return Promise.all(names.map((n) => {
        if (!this.pend[n]) {
          this.pend[n] = fetch('data/raster/' + n).then((r) => { if (!r.ok) throw new Error(n + ' ' + r.status); return r.arrayBuffer(); }).then((b) => {
            const ext = n.split('.').pop();
            this.raw[n] = ext === 'i16' ? new Int16Array(b) : ext === 'u16' ? new Uint16Array(b) : new Uint8Array(b);
          });
        }
        return this.pend[n];
      }));
    }
    async setK(k) {
      this.k = k; const token = (this._tok = (this._tok || 0) + 1);
      let cls;
      if (Math.abs(k - this.site.k) < 1e-9) { await this.ensure(['class_k2.u8']); cls = this.raw['class_k2.u8']; this.exact = true; }
      else {
        await this.ensure(['class_k2.u8', 'd.i16', 'sigma.i16', 'parent10.u8']);
        if (token !== this._tok) return;
        const d = this.raw['d.i16'], sg = this.raw['sigma.i16'], par = this.raw['parent10.u8'], base = this.raw['class_k2.u8'];
        cls = new Uint8Array(this.W * this.H); const W = this.W, w10 = this.w10;
        for (let r = 0; r < this.H; r++) {
          const pr = (r >> 2) * w10;
          for (let c = 0; c < W; c++) {
            const i = r * W + c;
            if (base[i] === 255) { cls[i] = 255; continue; }
            const s = d[i] > k * sg[i], p = par[pr + (c >> 2)] === 1;
            cls[i] = s && p ? 1 : p ? 2 : s ? 3 : 0;
          }
        }
        this.exact = false;
      }
      this.cls = cls;
      const n = { 0: 0, 1: 0, 2: 0, 3: 0, 255: 0 }; for (let i = 0; i < cls.length; i++) n[cls[i]]++;
      this.counts = n; this._paint(); this.emit('counts', { counts: n, k, exact: this.exact });
    }
    _paint() {
      if (!this.cls) return;
      const ctx = this.canvas.getContext('2d'), img = ctx.createImageData(this.W, this.H), px = img.data, cls = this.cls;
      for (let i = 0, j = 0; i < cls.length; i++, j += 4) {
        const c = cls[i]; if (c === 0 || !this.visible.has(c)) continue;
        const col = COLORS[c]; px[j] = col[0]; px[j + 1] = col[1]; px[j + 2] = col[2]; px[j + 3] = col[3];
      }
      ctx.putImageData(img, 0, 0);
    }
    // Everything the inspector needs for one 2.5 m pixel (loads the rasters on first use).
    async pixel(col, row) {
      await this.ensure(['d.i16', 'sigma.i16', 'parent10.u8', 'class_k2.u8', 'bands10.u16', 'parent_drop10.i16']);
      const i = row * this.W + col, i10 = (row >> 2) * this.w10 + (col >> 2), n10 = this.w10 * this.h10;
      const b = this.raw['bands10.u16'], pre = {}, post = {};
      BANDS.forEach((name, bi) => { pre[name] = b[bi * n10 + i10] / 1e4; post[name] = b[(4 + bi) * n10 + i10] / 1e4; });
      const nd = (x) => (x.B08 + x.B04) > 0 ? (x.B08 - x.B04) / (x.B08 + x.B04) : null;
      return {
        col, row, ...this.lonlat(col, row), cls: this.cls ? this.cls[i] : null, stored: this.raw['class_k2.u8'][i],
        d: this.raw['d.i16'][i] / 1e4, sigma: this.raw['sigma.i16'][i] / 1e4, parent: this.raw['parent10.u8'][i10] === 1,
        parentDrop: this.raw['parent_drop10.i16'][i10] / 1e4, col10: col >> 2, row10: row >> 2, pre, post, ndviPre: nd(pre), ndviPost: nd(post),
      };
    }
  }
  window.TSRViewer = { Viewer, CLS, COLORS, BANDS };
})();
