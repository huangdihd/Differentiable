import { expect, it } from 'vitest';
import { applyResult, initialProgress, evaluateLevel } from '../engine/levels';
import { gameLevels } from './catalog';
import { chapter3ClassificationLevels, chapter3ForwardLevels } from './chapter3';
import { referenceRegistry } from './level-helpers';
import { availableComponents, buildPlayerRegistry } from './registry';
it('earns the player cross_entropy subgraph with p/t ports', async () => {
  let progress = initialProgress(gameLevels.map((l) => l.id));
  for (const level of gameLevels.filter((l) => l.kind !== 'train')) {
    if (level.id === '3-7') break;
    const result = await evaluateLevel(
      { ...level, available: availableComponents(level, progress) },
      level.referenceGraph!,
      { registry: buildPlayerRegistry(gameLevels, progress) },
    );
    expect(result.passed, level.id).toBe(true);
    progress = applyResult(progress, level, result, level.referenceGraph!);
  }
  const component = buildPlayerRegistry(gameLevels, progress).get('cross_entropy')!;
  expect(component.inputs).toEqual(['p', 't']);
  expect(component.subgraph).toBeDefined();
  expect(component.native).toBeUndefined();
});
it('classification worker traces retain three steps, accuracy and output loss gradients', async () => {
  const { WorkerRuntime } = await import('../worker/runtime');
  const messages: import('../worker/protocol').WorkerResponse[] = [];
  const progress = initialProgress(
    gameLevels.map((l) => l.id),
    gameLevels.flatMap((l) => l.unlockComponents ?? []),
  );
  progress.passed = gameLevels.map((l) => l.id);
  progress.passedSolutions = Object.fromEntries(gameLevels.map((l) => [l.id, l.referenceGraph!]));
  const runtime = new WorkerRuntime((m) => messages.push(m), gameLevels, progress);
  let requestId = 0;
  try {
    await runtime.handle({ type: 'init', backend: 'cpu', revision: 0, requestId: ++requestId });
    for (const level of [
      ...chapter3ClassificationLevels,
      gameLevels.find((level) => level.id === 'xor')!,
    ]) {
      let last = Infinity;
      for (let i = 1; i <= 3; i++) {
        await runtime.handle({
          type: 'trainStep',
          levelId: level.id,
          graph: level.referenceGraph!,
          learningRate: 0.001,
          optimizer: level.id === '3-8' ? 'sgd' : 'adam',
          revision: 0,
          requestId: ++requestId,
        });
        const reply = messages.at(-1)!;
        expect(reply.type).toBe('reply');
        if (reply.type !== 'reply') throw new Error(JSON.stringify(reply));
        const trace = reply.data as import('../worker/protocol').ReplyData['trainStep'];
        expect(trace.step).toBe(i);
        expect(trace.accuracy).toBeTypeOf('number');
        expect(trace.gradient.output.values.some((v) => v !== 0)).toBe(true);
        // Adam need not decrease monotonically on every early step (notably fixed-rate XOR).
        // The simple two-cluster lesson is the three-step descent regression fixture.
        if (level.id === '3-6') expect(trace.loss).toBeLessThan(last);
        expect(Number.isFinite(trace.loss)).toBe(true);
        last = trace.loss;
      }
    }
  } finally {
    await runtime.dispose();
  }
});
const forwards = gameLevels.filter((l) => l.kind !== 'train');
const registry = referenceRegistry(forwards);
const playable = (level: (typeof gameLevels)[number]) => ({
  ...level,
  available: [...level.available, ...forwards.flatMap((l) => l.unlockComponents ?? [])],
});
for (const level of chapter3ClassificationLevels) {
  it(`${level.id} measured training settings`, async () => {
    const settings =
      level.id === '3-6'
        ? [{ learningRate: undefined }, { learningRate: 0.1 }]
        : level.id === '3-8'
          ? [0.001, 0.005, 0.01, 0.05, 0.1].map((learningRate) => ({
              learningRate,
              optimizer: 'sgd' as const,
            }))
          : [{ learningRate: undefined }];
    for (const setting of [
      ...settings,
      ...(level.id === '3-8' ? [{ optimizer: 'adam' as const }] : []),
    ]) {
      const result = await evaluateLevel(playable(level), level.referenceGraph!, {
        registry,
        ...setting,
      });
      console.info(level.id, 'par', level.par, setting, result.training, result.errorKey);
      const shouldPass =
        level.id === '3-9' ||
        ('optimizer' in setting && setting.optimizer === 'adam') ||
        (level.id === '3-6' && 'learningRate' in setting && setting.learningRate === 0.1);
      expect(result.passed).toBe(shouldPass);
    }
  }, 60000);
}
for (const level of chapter3ForwardLevels.slice(3)) {
  it(`${level.id} reference and mistakes`, async () => {
    const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(result.passed).toBe(true);
    if (level.id === '3-5') {
      expect(result.tests).toHaveLength(15);
      for (const mistake of ['sign', 'sum']) {
        const wrong = structuredClone(level.referenceGraph!);
        if (mistake === 'sign') wrong.nodes.find((n) => n.id === 'minus')!.attrs = { value: 1 };
        else
          Object.assign(
            wrong.nodes.find((n) => n.id === 'average')!,
            { component: 'sum', attrs: { axes: [-1], keepdims: true } },
          );
        expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
      }
    } else {
      const wrong = structuredClone(level.referenceGraph!);
      for (const edge of wrong.edges) {
        if (edge.from.node === 'm') edge.from.node = 'g';
        else if (edge.from.node === 'g') edge.from.node = 'm';
      }
      expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
    }
    console.info('par', level.id, level.par);
  });
}
