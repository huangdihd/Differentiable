import type { ComponentRegistry, GraphJSON, GraphNode } from '../engine/graph';
import { translate, translateDiagnostic, type Locale } from '../i18n';
import { parameterName } from './parameter-label';

type Diagnostic = Parameters<typeof translateDiagnostic>[1];

/** Expanded paths point at the existing outer canvas node; retain the full path in the error. */
export function diagnosticNode(graph: GraphJSON, error: Diagnostic): string | undefined {
  if (!error.node) return undefined;
  if (graph.nodes.some((node) => node.id === error.node)) return error.node;
  try {
    const id = decodeURIComponent(error.node.split('/')[0]!);
    return graph.nodes.find((node) => node.id === id)?.id;
  } catch {
    return undefined;
  }
}

/** Resolve the actual failing operation, not just its outer canvas container. */
function failingNode(
  graph: GraphJSON,
  path: string | undefined,
  registry?: ComponentRegistry,
): GraphNode | undefined {
  if (!path) return undefined;
  const exact = graph.nodes.find((node) => node.id === path);
  if (exact) return exact;
  try {
    const segments = path.split('/').map(decodeURIComponent);
    let current = graph;
    for (let index = 0; index < segments.length; index++) {
      const node = current.nodes.find((node) => node.id === segments[index]);
      if (!node) return undefined;
      if (index === segments.length - 1) return node;
      const subgraph = registry?.get(node.component)?.subgraph;
      if (!subgraph) return undefined;
      // Shape-dependent factories may not resolve without compile-time inputs.
      current = typeof subgraph === 'function' ? subgraph(node.attrs ?? {}) : subgraph;
    }
  } catch {
    // Malformed paths and unavailable generated subgraphs use generic advice.
  }
  return undefined;
}

export function graphDiagnostic(
  locale: Locale,
  graph: GraphJSON,
  error: Diagnostic,
  registry?: ComponentRegistry,
  named = true,
): string {
  const node = graph.nodes.find((node) => node.id === diagnosticNode(graph, error));
  let message = translateDiagnostic(locale, error);
  if ((error.key ?? error.message) === 'training.nonfinite' || error.params?.nonfinite === true) {
    if (error.params?.actual !== undefined)
      message += ` ${translate(locale, 'training.nonfinite.value', {
        actual: String(error.params.actual),
        index: Number(error.params.index ?? 0) + 1,
      })}`;
    const component = failingNode(graph, error.node, registry)?.component;
    const hint =
      component && ['log', 'exp', 'div', 'recip', 'sqrt', 'param'].includes(component)
        ? component
        : 'default';
    message += ` ${translate(locale, `training.nonfinite.hint.${hint}`)}`;
  }
  if (!node || !named) return message;
  const label = parameterName(node) ?? translate(locale, `components.${node.component}`);
  return `${label}：${message}`;
}
