// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { buildBackwardGraph, traceBackward } from './backward-graph';
import { CpuBackend } from './cpu';
import { compileGraph, type ComponentRegistry, type GraphJSON, type Attributes } from './graph';
import type { TrainLevel } from './levels';
import { Random } from './random';
import * as T from './tensor';

export type ParameterSnapshot = Map<string, { shape: number[]; data: Float32Array }>;

/** A read-only CPU view. Neither the player's graph nor training-owned tensors are mutated. */
export async function trainingBackward(
  level: TrainLevel,
  source: GraphJSON,
  registry: ComponentRegistry,
  saved?: ParameterSnapshot,
) {
  const batch = (level.resample ? level.evaluationDataset! : level.dataset)(new Random(level.seed));
  const backend = new CpuBackend();
  const inputs: Record<string, T.Tensor> = {};
  const parameters = new Map<string, T.Tensor>();
  let plan: ReturnType<typeof compileGraph> | undefined;
  try {
    for (const [name, value] of Object.entries(batch.inputs))
      inputs[name] = T.tensor(await value.read(), [...value.shape], backend);
    if (saved)
      for (const [id, value] of saved)
        parameters.set(id, T.tensor(value.data, value.shape, backend));
    const inputShapes = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v.shape]));
    plan = compileGraph(source, {
      registry,
      availableComponents: level.available,
      inputShapes,
      backend,
      random: new Random(level.seed),
      parameters,
    });
    // Use the exact training compiler's lazy initialization order, not the expanded graph's order.
    const outputs = plan.execute(inputs);
    plan.releaseExecution(outputs);
    const graph = structuredClone(source);
    const output = graph.nodes.find(
      (n) =>
        n.component === 'output' && n.attrs?.name === (level.lossOutput ?? level.interface.output),
    );
    if (!output) throw new Error('levels.output_shape');
    const attachedLoss = !level.lossOutput;
    if (attachedLoss) {
      const target = batch.targets;
      if (
        !target ||
        JSON.stringify(plan.outputShapes[level.interface.output]) !== JSON.stringify(target.shape)
      )
        throw new Error('levels.output_shape');
      const edge = graph.edges.find((e) => e.to.node === output.id && e.to.port === 'x')!;
      const prediction = edge.from;
      graph.edges = graph.edges.filter((e) => e !== edge);
      const used = new Set(graph.nodes.map((n) => n.id));
      const append = (
        name: string,
        component: string,
        args: Record<string, { node: string; port: string }>,
        attrs: Attributes = {},
      ) => {
        let id = `__loss_${name}`;
        while (used.has(id)) id += '_';
        used.add(id);
        graph.nodes.push({ id, component, attrs: { ...attrs, engineLoss: true } });
        for (const [port, from] of Object.entries(args))
          graph.edges.push({ from, to: { node: id, port } });
        return { node: id, port: 'out' };
      };
      const t = append(
        'targets',
        'const',
        {},
        { shape: [...target.shape], value: Array.from(await target.read()) },
      );
      const eps = append('epsilon', 'const', {}, { value: 1e-7 });
      const log = append('log', 'log', {
        x: append('stabilized', 'add', { a: prediction, b: eps }),
      });
      const product = append('product', 'mul', { a: t, b: log });
      const sum = append('sum', 'sum', { x: product }, { axes: target.shape.map((_, i) => i) });
      const rows = target.shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const scale = append('scale', 'const', {}, { value: -1 / rows });
      const loss = append('mean', 'mul', { a: sum, b: scale });
      graph.edges.push({ from: loss, to: { node: output.id, port: 'x' } });
    }
    output.attrs = { ...output.attrs, name: 'loss' };
    const wrt = [...plan.parameters.keys()].map((path) =>
      path.split('/').map(decodeURIComponent).join('/'),
    );
    const options = {
      registry,
      inputShapes,
      output: 'loss',
      wrt,
    };
    const backward = buildBackwardGraph(graph, options);
    const data: Record<string, number[]> = {};
    for (const [name, value] of Object.entries(inputs)) data[name] = Array.from(await value.read());
    const trace = await traceBackward(graph, {
      ...options,
      inputs: data,
      parameters,
      previewLimit: 16,
    });
    return {
      graph,
      backward,
      trace,
      attachedLoss,
      wrt,
      parameterState: saved ? ('current' as const) : ('initial' as const),
    };
  } finally {
    plan?.dispose();
    parameters.forEach((t) => t.dispose());
    Object.values(inputs).forEach((t) => t.dispose());
    new Set([...Object.values(batch.inputs), batch.targets]).forEach((t) => t?.dispose());
  }
}
