// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useRef } from 'react';

/** Training loss on a log scale, with the level's pass threshold as a dashed line. */
export default function LossChart({
  losses,
  threshold,
  label,
  thresholdLabel,
}: {
  losses: number[];
  threshold?: number;
  label: string;
  thresholdLabel: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const draw = () => {
      if (!canvas) return;
      const g = canvas.getContext('2d');
      if (!g) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const css = getComputedStyle(canvas);
      const color = (name: string) => css.getPropertyValue(name).trim();
      const mono = color('--mono');
      const pad = { l: 34, r: 8, t: 10, b: 18 };
      const values = losses.filter((v) => Number.isFinite(v) && v > 0);
      const all = threshold ? [...values, threshold] : values;
      const hi = Math.ceil(Math.log10(Math.max(...all, 1)));
      const lo = Math.floor(Math.log10(Math.min(...all, 0.1)));
      const sy = (v: number) =>
        pad.t + ((hi - Math.log10(Math.max(v, 10 ** lo))) / (hi - lo || 1)) * (h - pad.t - pad.b);
      const n = Math.max(values.length, 50);
      const sx = (i: number) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r);

      g.font = `10px ${mono}`;
      g.lineWidth = 1;
      for (let e = lo; e <= hi; e++) {
        g.strokeStyle = color('--line');
        g.beginPath();
        g.moveTo(pad.l, sy(10 ** e));
        g.lineTo(w - pad.r, sy(10 ** e));
        g.stroke();
        g.fillStyle = color('--faint');
        g.fillText(e === 0 ? '1' : `1e${e}`, 2, sy(10 ** e) + 3);
      }
      if (threshold) {
        g.setLineDash([4, 4]);
        g.strokeStyle = color('--good');
        g.beginPath();
        g.moveTo(pad.l, sy(threshold));
        g.lineTo(w - pad.r, sy(threshold));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = color('--good');
        g.textAlign = 'right';
        g.fillText(thresholdLabel, w - pad.r, sy(threshold) - 4);
        g.textAlign = 'left';
      }
      if (!values.length) return;
      g.strokeStyle = color('--accent');
      g.lineWidth = 1.8;
      g.beginPath();
      values.forEach((v, i) => (i ? g.lineTo(sx(i), sy(v)) : g.moveTo(sx(i), sy(v))));
      g.stroke();
      const last = values.length - 1;
      g.fillStyle = color('--accent');
      g.fillRect(sx(last) - 2.5, sy(values[last]) - 2.5, 5, 5);
    };
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas) observer.observe(canvas);
    return () => observer.disconnect();
  }, [losses, threshold, thresholdLabel]);
  return <canvas className="loss-chart" ref={ref} aria-label={label} role="img" />;
}
