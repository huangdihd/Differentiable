// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
// 主题切换：电路板 / 论文 / 蓝图。页面里放一个 #theme-switch 即可。
(function () {
  const THEMES = [['circuit', '电路板'], ['paper', '论文'], ['blueprint', '蓝图']];
  let current = 'blueprint';
  try { current = localStorage.getItem('diff-theme') || current; } catch (e) {}
  document.documentElement.dataset.theme = current;

  function mark() {
    document.querySelectorAll('#theme-switch button').forEach(b =>
      b.classList.toggle('on', b.dataset.id === document.documentElement.dataset.theme));
  }
  function set(id) {
    document.documentElement.dataset.theme = id;
    try { localStorage.setItem('diff-theme', id); } catch (e) {}
    mark();
    window.dispatchEvent(new Event('themechange'));
  }
  window.addEventListener('DOMContentLoaded', () => {
    const host = document.getElementById('theme-switch');
    if (!host) return;
    host.classList.add('theme-switch');
    for (const [id, name] of THEMES) {
      const b = document.createElement('button');
      b.textContent = name; b.dataset.id = id; b.onclick = () => set(id);
      host.appendChild(b);
    }
    mark();
  });

  window.cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  window.hexRgb = hex => {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  // 高 DPI canvas
  window.fitCanvas = c => {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  };
})();
