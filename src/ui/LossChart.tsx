import { useEffect, useRef } from 'react';
export default function LossChart({ losses, label }: { losses: number[]; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, 600, 280);
    ctx.strokeStyle = getComputedStyle(canvas).getPropertyValue('--accent');
    ctx.lineWidth = 2;
    const values = losses.filter(Number.isFinite);
    const max = Math.max(...values, 0.001);
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i * 600) / Math.max(1, values.length - 1),
        y = 275 - (v / max) * 270;
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
  }, [losses]);
  return (
    <canvas
      className="loss-chart"
      ref={ref}
      width={600}
      height={280}
      aria-label={label}
      role="img"
    />
  );
}
