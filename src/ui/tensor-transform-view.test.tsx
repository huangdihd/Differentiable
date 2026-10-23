import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import TensorTransformView, {
  explainTensorTransform,
  tensorTransformKind,
} from './TensorTransformView';

const close = (actual: number[], expected: number[]) =>
  actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 7));

describe('tensor transform visualizations', () => {
  it('shows stable Softmax stages without exponentiating the original large scores', () => {
    const result = explainTensorTransform('softmax', [100, 101, 102]);
    expect(result.stages.map((stage) => stage.key)).toEqual([
      'input',
      'shifted',
      'weights',
      'output',
    ]);
    expect(result.stages[1].values).toEqual([-2, -1, 0]);
    close(result.stages[2].values, [Math.exp(-2), Math.exp(-1), 1]);
    expect(result.stages[3].values.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 7);
    expect(result.stages.flatMap((stage) => stage.values).every(Number.isFinite)).toBe(true);
  });

  it('shows LayerNorm centering, variance and normalized output', () => {
    const result = explainTensorTransform('layernorm', [1, 2, 3]);
    expect(result.stages.map((stage) => stage.key)).toEqual([
      'input',
      'centered',
      'squares',
      'output',
    ]);
    expect(result.stages[1].values).toEqual([-1, 0, 1]);
    expect(result.stats.find((stat) => stat.key === 'mean')?.value).toBe(2);
    expect(result.stats.find((stat) => stat.key === 'variance')?.value).toBeCloseTo(2 / 3, 7);
    const output = result.stages[3].values;
    expect(output.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 7);
  });

  it('shows MSE as prediction, target, error, squared loss, then last-axis Mean', () => {
    const result = explainTensorTransform('mse', [1, 4], [3, 1]);
    expect(result.stages.map((stage) => stage.key)).toEqual([
      'prediction',
      'target',
      'errors',
      'squares',
      'output',
    ]);
    expect(result.stages.map((stage) => stage.values)).toEqual([
      [1, 4],
      [3, 1],
      [-2, 3],
      [4, 9],
      [6.5],
    ]);
    expect(result.stats).toEqual([{ key: 'mean', value: 6.5 }]);
    expect(explainTensorTransform('mse', [1, 2], [1]).stages).toEqual([]);
  });

  it('shows cross entropy selecting one-hot log probabilities and averaging row losses', () => {
    const result = explainTensorTransform('cross_entropy', [0.2, 0.7, 0.1], [0, 1, 0]);
    expect(result.stages.map((stage) => stage.key)).toEqual([
      'probabilities',
      'logs',
      'target',
      'selected',
      'rowLoss',
    ]);
    close(result.stages[1].values, [Math.log(0.2), Math.log(0.7), Math.log(0.1)]);
    expect(result.stages[2].values).toEqual([0, 1, 0]);
    close(result.stages[3].values, [0, Math.log(0.7), 0]);
    expect(result.stages[4].values[0]).toBeCloseTo(-Math.log(0.7), 7);
    expect(result.stats).toEqual([{ key: 'rowLoss', value: -Math.log(0.7) }]);
    expect(explainTensorTransform('cross_entropy', [0.2, 0.8], [1]).stages).toEqual([]);
  });

  it('selects only the intended lesson visualizations and renders every row stage', () => {
    expect(tensorTransformKind('2-7')).toBe('softmax');
    expect(tensorTransformKind('2-8')).toBe('layernorm');
    expect(tensorTransformKind('3-3')).toBe('mse');
    expect(tensorTransformKind('3-5')).toBe('cross_entropy');
    expect(tensorTransformKind('2-6')).toBeUndefined();
    const html = renderToStaticMarkup(
      <TensorTransformView
        kind="softmax"
        values={[1, 2, 3, 4]}
        shape={[2, 2]}
        t={(key, params) => `${key}${params ? JSON.stringify(params) : ''}`}
        format={String}
      />,
    );
    expect(html).toContain('visual.softmax.title');
    expect(html).toContain('visual.row{&quot;current&quot;:1,&quot;total&quot;:2}');
    expect(html.match(/class="tensor-transform-stage"/g)).toHaveLength(4);
    expect(html).toContain('visual.softmax.output');

    const mse = renderToStaticMarkup(
      <TensorTransformView
        kind="mse"
        values={[1, 4, 2, 6]}
        targets={[3, 1, 1, 4]}
        shape={[2, 2]}
        t={(key, params) => `${key}${params ? JSON.stringify(params) : ''}`}
        format={String}
      />,
    );
    expect(mse).toContain('visual.mse.title');
    expect(mse).toContain('visual.row{&quot;current&quot;:1,&quot;total&quot;:2}');
    expect(mse.match(/class="tensor-transform-stage"/g)).toHaveLength(5);
    expect(mse).toContain('visual.mse.output');

    const crossEntropy = renderToStaticMarkup(
      <TensorTransformView
        kind="cross_entropy"
        values={[0.2, 0.7, 0.1, 0.6, 0.3, 0.1]}
        targets={[0, 1, 0, 1, 0, 0]}
        shape={[2, 3]}
        t={(key, params) => `${key}${params ? JSON.stringify(params) : ''}`}
        format={String}
      />,
    );
    expect(crossEntropy).toContain('visual.cross_entropy.title');
    expect(crossEntropy).toContain('visual.row{&quot;current&quot;:1,&quot;total&quot;:2}');
    expect(crossEntropy.match(/class="tensor-transform-stage"/g)).toHaveLength(5);
    expect(crossEntropy).toContain('visual.cross_entropy.rowLoss');
    expect(crossEntropy).toContain('visual.cross_entropy.batchMean');
    expect(crossEntropy).toContain(String((-Math.log(0.7) - Math.log(0.6)) / 2));
  });
});
