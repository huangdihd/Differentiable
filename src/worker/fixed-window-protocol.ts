import type { Progress } from '../engine/levels';
import type { WorkerError } from './protocol';

export type CharacterComparisonModel = 'fixed-window' | 'rnn';

export interface FixedWindowPrediction {
  history: string;
  input: string;
  probabilities: number[];
  prediction: string;
}

export interface FixedWindowSnapshot {
  model: CharacterComparisonModel;
  parameterCount: number;
  sampleCount: number;
  step: number;
  loss: number;
  stepsPerRun: number;
  seed: number;
  learningRate: number;
  predictions: FixedWindowPrediction[];
  resetPredictions?: FixedWindowPrediction[];
}

export type FixedWindowRequest = {
  type: 'train';
  model: CharacterComparisonModel;
  /** Selected by the main game; only the first request creates the comparison backend. */
  backend?: 'cpu' | 'webgpu';
  /** Supplied only when creating a post-build experiment; later runs retain the same circuit. */
  playerProgress?: Progress;
};

export type FixedWindowResponse =
  | { type: 'progress'; step: number; totalSteps: number; loss?: number }
  | { type: 'result'; snapshot: FixedWindowSnapshot }
  | ({ type: 'error'; key: string } & Partial<Omit<WorkerError, 'key'>>);
