import { evaluateLevel, type CircuitLevel, type LevelResult, type GraphMetrics } from './levels';
import type { GraphJSON, ComponentRegistry } from './graph';

/** A level may combine any number of independently seeded forward, derivative,
 * and training checks. Every check starts with fresh parameter initialization. */
export interface LevelSuite {
  id: string;
  chapter: number;
  titleKey: string;
  descriptionKey: string;
  hintKeys: readonly string[];
  unlockComponents: readonly string[];
  tests: readonly CircuitLevel[];
}
export interface SuiteResult {
  levelId: string;
  passed: boolean;
  tests: LevelResult[];
  metrics?: GraphMetrics;
}
export async function evaluateSuite(
  level: LevelSuite,
  graph: GraphJSON,
  options: { registry?: ComponentRegistry } = {},
): Promise<SuiteResult> {
  const tests: LevelResult[] = [];
  for (const test of level.tests) tests.push(await evaluateLevel(test, graph, options));
  return {
    levelId: level.id,
    passed: tests.length > 0 && tests.every((t) => t.passed),
    tests,
    metrics: tests[0]?.metrics,
  };
}
