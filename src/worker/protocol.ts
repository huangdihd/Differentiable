import type { buildBackwardGraph, traceBackward } from '../engine/backward-graph';
import type { GraphEndpoint, GraphJSON, ShapePorts } from '../engine/graph';
import type { GraphMetrics, LevelResult, Progress } from '../engine/levels';
import type { OptimizerOptions, TrainingReport, TrainingSession } from '../engine/training';

export interface WorkerError {
  key: string;
  params: Record<string, unknown>;
  node?: string;
  port?: string;
}
export type BackendPreference = 'auto' | 'cpu' | 'webgpu';
export type Command =
  | { type: 'init'; backend?: BackendPreference }
  | { type: 'syncProgress'; progress: Progress }
  | { type: 'compile'; levelId: string; graph: GraphJSON }
  | {
      type: 'probe';
      levelId: string;
      graph: GraphJSON;
      endpoint: GraphEndpoint;
      inputs: Record<string, number[]>;
      /** One of the level's shape groups (main or a variant); defaults to the main shapes. */
      shapes?: Record<string, number[]>;
    }
  | {
      type: 'backward';
      levelId: string;
      graph: GraphJSON;
      inputs?: Record<string, number[]>;
    }
  | {
      type: 'evaluate';
      levelId: string;
      graph: GraphJSON;
      learningRate?: number;
      optimizer?: OptimizerOptions['kind'];
    }
  | {
      type: 'start';
      levelId: string;
      graph: GraphJSON;
      learningRate?: number;
      optimizer?: OptimizerOptions['kind'];
    }
  | {
      type: 'trainStep';
      levelId: string;
      graph: GraphJSON;
      learningRate?: number;
      optimizer?: OptimizerOptions['kind'];
    }
  | {
      type: 'generate';
      levelId: string;
      graph: GraphJSON;
      prompt: string;
      length: number;
      seed?: number;
      temperature?: number;
    }
  | { type: 'trainReset' }
  | { type: 'cancel' }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'stop' };
export type WorkerRequest = Command & { requestId: number; revision: number };
export interface ReplyData {
  init: { backend: 'cpu' | 'webgpu'; fallback?: WorkerError };
  syncProgress: { state: 'synced' };
  compile: { outputShapes: ShapePorts; metrics: GraphMetrics; diagnostics: WorkerError[] };
  probe: { shape: readonly number[]; values: number[] };
  backward: {
    backward: ReturnType<typeof buildBackwardGraph>;
    trace?: Awaited<ReturnType<typeof traceBackward>>;
  };
  evaluate: LevelResult;
  start: { state: 'running' };
  trainStep: Awaited<ReturnType<TrainingSession['stepTrace']>>;
  generate: { text: string };
  trainReset: { state: 'reset' };
  cancel: { state: 'stopped' };
  pause: { state: 'paused' };
  resume: { state: 'running' };
  stop: { state: 'stopped' };
}
export type WorkerResponse =
  | {
      type: 'reply';
      requestId: number;
      revision: number;
      command: Command['type'];
      data: ReplyData[Command['type']];
    }
  | { type: 'error'; requestId: number; revision: number; error: WorkerError }
  | { type: 'progress'; requestId: number; revision: number; report: TrainingReport }
  | { type: 'result'; requestId: number; revision: number; result: LevelResult };
