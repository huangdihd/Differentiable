import { expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { Random } from '../engine/random';
import { crossEntropy, TrainingSession } from '../engine/training';
import { createInitialProgress } from '../storage';
import { gameLevels } from './catalog';
import { attentionConfig, chapter5TrainLevels } from './chapter5';
import { buildPlayerRegistry } from './registry';

/** Progress as if every level before 5-5 was passed (unlocks embed_pos, attention, linear, …). */
function unlockBefore55() {
  let p = createInitialProgress();
  for (const l of gameLevels) {
    if (l.id === '5-5') break;
    p = {
      ...p,
      passed: [...new Set([...p.passed, l.id])],
      unlocked: [...new Set([...p.unlocked, ...l.unlock, l.id])],
      unlockedComponents: [...new Set([...p.unlockedComponents, ...(l.unlockComponents ?? [])])],
    };
  }
  return p;
}

/**
 * Graph described by levels.5-5.hint1/hint2: embed_pos → three Linear 12→12 → attention →
 * Linear 12→27 → Softmax. Must beat the train threshold so the hints stay buildable.
 */
function hintedGraph(): GraphJSON {
  const g: GraphJSON = { version: 1, nodes: [], edges: [] };
  const node = (
    id: string,
    component: string,
    args: Record<string, string> = {},
    attrs?: Record<string, unknown>,
  ) => {
    g.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, source] of Object.entries(args))
      g.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  node('x', 'input', {}, { name: 'x' });
  const emb = node(
    'emb',
    'embed_pos',
    { idx: 'x' },
    { automatic_parameters: true, num_embeddings: 27, max_length: 8, embedding_dim: 12 },
  );
  const lin = (id: string) =>
    node(
      id,
      'linear',
      { x: emb },
      { automatic_parameters: true, in_features: 12, out_features: 12, bias: true },
    );
  const att = node('att', 'attention', { q: lin('q'), k: lin('k'), v: lin('v') });
  node(
    'logits',
    'linear',
    { x: att },
    { automatic_parameters: true, in_features: 12, out_features: 27, bias: true },
  );
  node('probability', 'softmax', { x: 'logits' });
  node('output', 'output', { x: 'probability' }, { name: 'probabilities' });
  return g;
}

it('5-5 hint recipe (embed_pos + linear q/k/v + attention + linear + Softmax) beats threshold', async () => {
  const level = chapter5TrainLevels[0];
  const registry = buildPlayerRegistry(gameLevels, unlockBefore55());
  for (const name of ['embed_pos', 'linear', 'attention', 'softmax'])
    expect(registry.get(name), name).toBeTruthy();
  const plan = compileGraph(hintedGraph(), {
    registry,
    random: new Random(level.seed),
    inputShapes: level.interface.inputs,
  });
  const batch = level.dataset(new Random(level.seed));
  const evaluation = level.evaluationDataset!(new Random(level.seed + 1));
  const session = new TrainingSession(
    plan,
    batch,
    crossEntropy,
    level.optimizer,
    level.interface.output,
    { evaluationBatch: evaluation },
  );
  try {
    const final = await session.run(level.steps, { yieldEvery: 1000 });
    expect(final.loss).toBeLessThan(level.threshold.loss);
  } finally {
    session.dispose();
    plan.dispose();
    for (const data of [batch, evaluation]) {
      Object.values(data.inputs).forEach((t) => t.dispose());
      data.targets?.dispose();
    }
  }
}, 90000);

// Keep config in the failure message if the threshold ever drifts.
it('documents attentionConfig used by the hint recipe', () => {
  expect(attentionConfig).toEqual({ N: 128, T: 8, D: 12 });
});
