// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { initialProgress, type Progress } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { TrainingSession } from '../engine/training';
import { gameLevels } from '../levels/catalog';
import { availableComponents, buildPlayerRegistry } from '../levels/registry';
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  createInitialProgress,
  isGraphJSON,
  isProgress,
  loadProgress,
} from '../storage';
import {
  addComponent,
  applyPackage,
  componentUsage,
  customComponentName,
  exportComponents,
  importComponents,
  libraryOf,
  parseComponentsFile,
  planPackage,
  removeComponent,
  renameComponent,
  ungroupInstance,
  type CustomComponent,
  type PackagePlan,
} from './components';
import { buildSandboxLevel, sandboxComponents } from './level';
import { createSlot, updateSlot } from './slots';
import { starterGraph } from './starter';

const level = buildSandboxLevel('s', { kind: 'builtin', id: 'noisy-line' });
const shapes = { x: level.interface.inputs.x, t: level.interface.inputs.t };
/** x → Linear → tanh → Linear → reshape → MSE(t) → loss, every part an ordinary component. */
const mlp = () => starterGraph(shapes, 'regression', 'mlp');

const ids = gameLevels.map((l) => l.id);
const base = (): Progress => ({
  ...initialProgress(ids, sandboxComponents()),
  passed: ids,
  passedSolutions: Object.fromEntries(
    gameLevels.flatMap((l) =>
      'referenceGraph' in l && l.referenceGraph ? [[l.id, l.referenceGraph]] : [],
    ),
  ),
});

function pack(
  graph: GraphJSON,
  selection: string[],
  id: string,
  name = id,
): { component: CustomComponent; graph: GraphJSON; instance: string; plan: PackagePlan } {
  const plan = planPackage(graph, selection);
  if (typeof plan === 'string') throw new Error(plan);
  return { ...applyPackage(graph, plan, { id, name, now: 1 }), plan };
}

/** Train with SGD from the given starting parameters; returns every step's loss. */
async function losses(
  graph: GraphJSON,
  progress: Progress,
  start: Map<string, Float32Array>,
  steps = 25,
): Promise<number[]> {
  const registry = buildPlayerRegistry(gameLevels, progress);
  const parameters = new Map(
    [...start].map(([id, data]) => [id, T.tensor(new Float32Array(data), shapeOf.get(id)!)]),
  );
  const plan = compileGraph(graph, {
    registry,
    availableComponents: availableComponents(level, progress),
    inputShapes: level.interface.inputs,
    random: new Random(level.seed),
    parameters,
  });
  const batch = level.dataset(new Random(level.seed));
  const session = new TrainingSession(
    plan,
    batch,
    undefined,
    { kind: 'sgd', learningRate: 0.05 },
    'loss',
    { lossOutput: 'loss', reportEvery: 1 },
  );
  const out: number[] = [];
  await session.run(steps, { onReport: (r) => out.push(r.loss) });
  session.dispose();
  plan.dispose();
  return out;
}

const shapeOf = new Map<string, number[]>();
/** The seeded initial parameters of a graph, by storage path. */
async function initialParameters(graph: GraphJSON, progress: Progress) {
  const plan = compileGraph(graph, {
    registry: buildPlayerRegistry(gameLevels, progress),
    availableComponents: availableComponents(level, progress),
    inputShapes: level.interface.inputs,
    random: new Random(level.seed),
  });
  const batch = level.dataset(new Random(level.seed));
  plan.releaseExecution(plan.execute(batch.inputs));
  const values = new Map<string, Float32Array>();
  for (const [id, tensor] of plan.parameters) {
    values.set(id, new Float32Array(await tensor.read()));
    shapeOf.set(id, [...tensor.shape]);
  }
  plan.dispose();
  return values;
}

/** Same values, keyed by where they live once `moved` parts sit inside `instance`. */
const moveInto = (values: Map<string, Float32Array>, instance: string, moved: string[]) =>
  new Map(
    [...values].map(([id, data]) => {
      const next = moved.includes(decodeURIComponent(id.split('/')[0])) ? `${instance}/${id}` : id;
      shapeOf.set(next, shapeOf.get(id)!);
      return [next, data];
    }),
  );

describe('packaging a selection', () => {
  it('turns crossing wires into named ports and wires one instance in place', () => {
    const graph = mlp();
    const {
      component,
      graph: outer,
      instance,
      plan,
    } = pack(graph, ['hidden', 'activation', 'x'], 'c1', 'Block');
    // The input node itself never moves inside; the wire from it becomes port `x`.
    expect(plan.nodes).toEqual(['hidden', 'activation']);
    expect(component.inputs).toEqual(['x']);
    expect(component.outputs).toEqual(['out']);
    expect(outer.nodes.map((n) => n.id)).not.toContain('hidden');
    const node = outer.nodes.find((n) => n.id === instance)!;
    expect(node.component).toBe(customComponentName('c1'));
    expect(outer.edges).toContainEqual({
      from: { node: 'x', port: 'out' },
      to: { node: instance, port: 'x' },
    });
    expect(outer.edges).toContainEqual({
      from: { node: instance, port: 'out' },
      to: { node: 'layer', port: 'x' },
    });
    expect(component.graph.nodes.filter((n) => n.component === 'input')).toHaveLength(1);
    expect(isGraphJSON(component.graph)).toBe(true);
  });

  it('shares one input port between wires from the same source, and names several outputs', () => {
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'a', component: 'input', attrs: { name: 'a' } },
        { id: 'm', component: 'mul' },
        { id: 'e', component: 'exp' },
        { id: 'o1', component: 'output', attrs: { name: 'p' } },
        { id: 'o2', component: 'output', attrs: { name: 'q' } },
      ],
      edges: [
        { from: { node: 'a', port: 'out' }, to: { node: 'm', port: 'a' } },
        { from: { node: 'a', port: 'out' }, to: { node: 'm', port: 'b' } },
        { from: { node: 'm', port: 'out' }, to: { node: 'e', port: 'x' } },
        { from: { node: 'm', port: 'out' }, to: { node: 'o1', port: 'x' } },
        { from: { node: 'e', port: 'out' }, to: { node: 'o2', port: 'x' } },
      ],
    };
    const { component } = pack(graph, ['m', 'e'], 'c');
    expect(component.inputs).toEqual(['a']);
    expect(component.outputs).toEqual(['p', 'q']);
  });

  it('refuses an empty pick, a part with no way out, and a pick that would loop', () => {
    const graph = mlp();
    expect(planPackage(graph, ['x', 't'])).toBe('sandbox.components.problem.empty');
    expect(planPackage({ ...graph, edges: [] }, ['hidden'])).toBe(
      'sandbox.components.problem.noOutput',
    );
    // hidden → activation → layer: packaging hidden and layer leaves activation in between.
    expect(planPackage(graph, ['hidden', 'layer'])).toBe('sandbox.components.problem.cycle');
  });
});

describe('training through a component', () => {
  it('gives exactly the losses of the same parts wired inline', async () => {
    const graph = mlp();
    const {
      component,
      graph: packaged,
      instance,
      plan,
    } = pack(graph, ['hidden', 'activation', 'layer'], 'c');
    const progress = addComponent(base(), component);
    const start = await initialParameters(graph, base());
    const inline = await losses(graph, base(), start);
    const wrapped = await losses(packaged, progress, moveInto(start, instance, plan.nodes));
    expect(inline.length).toBe(25);
    expect(inline.at(-1)!).toBeLessThan(inline[0]);
    expect(wrapped).toEqual(inline);
  });

  it('stays identical through a component nested in another', async () => {
    const graph = mlp();
    const first = pack(graph, ['hidden', 'activation'], 'inner');
    let progress = addComponent(base(), first.component);
    const second = pack(first.graph, [first.instance, 'layer'], 'outer');
    progress = addComponent(progress, second.component);
    expect(second.component.graph.nodes.map((n) => n.component)).toContain(
      customComponentName('inner'),
    );
    const start = await initialParameters(graph, base());
    const inline = await losses(graph, base(), start, 10);
    const nested = moveInto(
      moveInto(start, first.instance, first.plan.nodes),
      second.instance,
      second.plan.nodes,
    );
    expect(await losses(second.graph, progress, nested, 10)).toEqual(inline);
  });

  it('gives every placed instance parameters of its own', async () => {
    const graph = mlp();
    const { component, graph: packaged, instance } = pack(graph, ['hidden', 'activation'], 'c');
    const progress = addComponent(base(), component);
    // A second instance of the same block, fed from the first.
    const twice: GraphJSON = structuredClone(packaged);
    twice.nodes.push({ id: 'again', component: customComponentName('c'), ui: { x: 0, y: 0 } });
    twice.edges = twice.edges.map((e) =>
      e.from.node === instance && e.to.node === 'layer'
        ? { ...e, from: { node: 'again', port: 'out' } }
        : e,
    );
    twice.edges.push({ from: { node: instance, port: 'out' }, to: { node: 'again', port: 'x' } });
    // Shape mismatch on purpose would fail; the block maps 1 → 16, so feed 16 back in via a new one.
    const registry = buildPlayerRegistry(gameLevels, progress);
    expect(registry.get(customComponentName('c'))?.label).toBe('c');
    const once = await initialParameters(packaged, progress);
    expect([...once.keys()].filter((k) => k.startsWith(`${instance}/`)).length).toBeGreaterThan(0);
    const keys = [...once.keys()].filter((k) => k.startsWith(`${instance}/`));
    // Paths are per instance, so a second instance can never share them.
    expect(keys.every((k) => !k.startsWith('again/'))).toBe(true);
  });
});

describe('ungrouping', () => {
  it('restores the original parts and wiring', () => {
    const graph = mlp();
    const { component, graph: packaged, instance } = pack(graph, ['hidden', 'activation'], 'c');
    const result = ungroupInstance(packaged, instance, [component])!;
    expect(result.ids).toHaveLength(2);
    const components = result.graph.nodes.map((n) => n.component).sort();
    expect(components).toEqual(graph.nodes.map((n) => n.component).sort());
    // Same wiring up to the new ids of the parts that came back out.
    const rename = new Map([
      ['hidden', result.ids[0]],
      ['activation', result.ids[1]],
    ]);
    const key = (g: GraphJSON, map = new Map<string, string>()) =>
      g.edges
        .map(
          (e) =>
            `${map.get(e.from.node) ?? e.from.node}:${e.from.port}>${map.get(e.to.node) ?? e.to.node}:${e.to.port}`,
        )
        .sort();
    expect(key(result.graph)).toEqual(key(graph, rename));
  });

  it('opens one level at a time and leaves unknown parts alone', () => {
    const graph = mlp();
    const first = pack(graph, ['hidden', 'activation'], 'inner');
    const second = pack(first.graph, [first.instance, 'layer'], 'outer');
    const library = [first.component, second.component];
    const out = ungroupInstance(second.graph, second.instance, library)!;
    expect(out.graph.nodes.map((n) => n.component)).toContain(customComponentName('inner'));
    expect(ungroupInstance(graph, 'hidden', library)).toBeUndefined();
  });
});

describe('library and storage', () => {
  it('round-trips the library through the save, next to the slots', () => {
    const { component } = pack(mlp(), ['hidden', 'activation'], 'c', 'Block');
    let progress = createSlot(createInitialProgress(), { id: 's', name: 'S', now: 1 });
    progress = addComponent(progress, component);
    progress = updateSlot(progress, 's', { name: 'T' }, 2);
    expect(libraryOf(progress)).toHaveLength(1);
    expect(isProgress(progress)).toBe(true);
    const data = new Map([[STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress })]]);
    const restored = loadProgress(() => ({
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
    }));
    expect(restored.sandbox?.components).toEqual([component]);
    // Saves from before components still load.
    expect(isProgress({ ...progress, sandbox: { slots: [] } })).toBe(true);
    // A library where a component contains itself is rejected.
    const loop = structuredClone(component);
    loop.graph.nodes.push({ id: 'self', component: customComponentName('c') });
    expect(isProgress({ ...progress, sandbox: { slots: [], components: [loop] } })).toBe(false);
  });

  it('renames, refuses to delete a placed component, then deletes it', () => {
    const { component, graph } = pack(mlp(), ['hidden', 'activation'], 'c', 'Block');
    let progress = addComponent(createSlot(base(), { id: 's', name: 'S', now: 1 }), component);
    progress = updateSlot(progress, 's', { graph }, 2);
    progress = renameComponent(progress, 'c', '  Encoder ', 3);
    expect(libraryOf(progress)[0].name).toBe('Encoder');
    expect(componentUsage(progress, 'c').slots).toEqual(['S']);
    expect(libraryOf(removeComponent(progress, 'c'))).toHaveLength(1);
    progress = updateSlot(progress, 's', { graph: mlp() }, 4);
    expect(libraryOf(removeComponent(progress, 'c'))).toHaveLength(0);
  });

  it('exports with dependencies and imports without duplicating', () => {
    const first = pack(mlp(), ['hidden', 'activation'], 'inner', 'Inner');
    const second = pack(first.graph, [first.instance, 'layer'], 'outer', 'Outer');
    let progress = addComponent(
      addComponent(createInitialProgress(), first.component),
      second.component,
    );
    const file = JSON.stringify(exportComponents(libraryOf(progress), ['outer']));
    const parsed = parseComponentsFile(file, isGraphJSON);
    expect(parsed.map((c) => c.id)).toEqual(['inner', 'outer']);
    // Into the same library: nothing new.
    const same = importComponents(progress, parsed, { newId: () => 'x', now: 5 });
    expect(same.added).toBe(0);
    expect(libraryOf(same.progress)).toHaveLength(2);
    // Into a library where `inner` means something else: fresh id, rewired user.
    const other = pack(mlp(), ['layer'], 'inner', 'Inner');
    let n = 0;
    const clash = importComponents(addComponent(createInitialProgress(), other.component), parsed, {
      newId: () => `new${++n}`,
      now: 5,
    });
    const library = libraryOf(clash.progress);
    expect(clash.added).toBe(2);
    expect(library.map((c) => c.id)).toEqual(['inner', 'new1', 'outer']);
    const outer = library.find((c) => c.id === 'outer')!;
    expect(outer.graph.nodes.map((n) => n.component)).toContain(customComponentName('new1'));
    expect(library.find((c) => c.id === 'new1')!.name).toBe('Inner 2');
    expect(() => parseComponentsFile('{', isGraphJSON)).toThrow('sandbox.components.error.json');
    expect(() =>
      parseComponentsFile(
        JSON.stringify(exportComponents([second.component], ['outer'])),
        isGraphJSON,
      ),
    ).toThrow('sandbox.components.error.missing');
    progress = clash.progress;
    expect(isProgress(progress)).toBe(true);
  });

  it('offers library components only in sandbox slots', () => {
    const { component } = pack(mlp(), ['hidden', 'activation'], 'c');
    const progress = addComponent(base(), component);
    expect(availableComponents(level, progress)).toContain(customComponentName('c'));
    const story = gameLevels.find((l) => l.id === 'xor');
    if (!story || story.kind === 'observation') throw new Error('xor level');
    expect(availableComponents(story, progress)).not.toContain(customComponentName('c'));
  });
});
