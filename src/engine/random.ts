// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Reproducible Mulberry32; never delegates to Math.random. */
export class Random {
  constructor(private state = 1) {}
  uniform(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal(): number {
    return (
      Math.sqrt(-2 * Math.log(Math.max(this.uniform(), 1e-12))) *
      Math.cos(2 * Math.PI * this.uniform())
    );
  }
}
