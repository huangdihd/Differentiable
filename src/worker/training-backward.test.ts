import { expect, it } from 'vitest';
import { initialProgress } from '../engine/levels';
import { gameLevels } from '../levels/catalog';
import { referenceRegistry } from '../levels/level-helpers';
import { WorkerRuntime } from './runtime';
import type { Command, ReplyData, WorkerResponse } from './protocol';

it('training backward uses current step parameters, resets, and guards expensive chapters', async () => {
  const preceding = gameLevels.filter((l) => l.kind !== 'train');
  const progress = initialProgress([]);
  progress.unlockedComponents = preceding.flatMap((l) => l.unlockComponents ?? []);
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(m),
    gameLevels,
    progress,
    referenceRegistry(preceding),
  );
  let requestId = 0;
  const send = async (command: Command) => {
    await runtime.handle({ ...command, revision: 0, requestId: ++requestId });
    return messages.at(-1)!;
  };
  const level = gameLevels.find((l) => l.id === '3-4')!;
  const graph = level.referenceGraph!;
  const backward = async () => {
    const reply = await send({ type: 'backward', levelId: level.id, graph });
    expect(reply).toMatchObject({ type: 'reply', command: 'backward' });
    return (reply as Extract<WorkerResponse, { type: 'reply' }>).data as ReplyData['backward'];
  };
  try {
    await send({ type: 'init', backend: 'cpu' });
    const first = await backward();
    expect(first.parameterState).toBe('initial');
    expect(await send({ type: 'trainStep', levelId: level.id, graph })).toMatchObject({
      type: 'reply',
    });
    const current = await backward();
    expect(current.parameterState).toBe('current');
    expect(current.trace!.forward.w.values).not.toEqual(first.trace!.forward.w.values);
    await send({ type: 'trainReset' });
    expect((await backward()).trace).toEqual(first.trace);
    expect(
      await send({
        type: 'backward',
        levelId: '6-4',
        graph: gameLevels.find((l) => l.id === '6-4')!.referenceGraph!,
      }),
    ).toMatchObject({ type: 'error', error: { key: 'worker.training_gradient_unavailable' } });
  } finally {
    await runtime.dispose();
  }
});
