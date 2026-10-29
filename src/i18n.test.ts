import { describe, expect, it, vi } from 'vitest';
import { createDefaultRegistry } from './engine/graph';
import { exampleLevels } from './levels/examples';
import { gameLevels } from './levels/catalog';
import {
  createTranslator,
  diagnosticKey,
  formatNumber,
  locales,
  localeNames,
  matchLocale,
  translate,
  translateDiagnostic,
  translatePort,
  translateValueName,
} from './i18n';

describe('i18n', () => {
  it('never exposes rank-zero empty brackets in translated player-facing copy', () => {
    for (const [locale, dictionary] of Object.entries(locales)) {
      for (const [key, message] of Object.entries(dictionary)) {
        const variants = typeof message === 'string' ? [message] : Object.values(message);
        for (const text of variants) expect(text, `${locale}: ${key}`).not.toContain('[]');
      }
    }
  });
  it('uses the canonical component namespace without paired aliases', () => {
    for (const dictionary of Object.values(locales)) {
      for (const key of Object.keys(dictionary)) {
        expect(key).not.toMatch(/^component\./);
        if (key.startsWith('components.'))
          expect(dictionary).not.toHaveProperty(key.replace('components.', 'component.'));
      }
    }
  });
  it('localizes semantic value names and preserves mathematical or custom identifiers', () => {
    expect(translateValueName(createTranslator('en'), 'probabilities')).toBe('Probabilities');
    expect(translateValueName(createTranslator('zh-CN'), 'probabilities')).toBe('概率');
    expect(translateValueName(createTranslator('zh-CN'), 'rowSums')).toBe('行和');
    expect(translateValueName(createTranslator('zh-CN'), 'f′')).toBe('f′');
    expect(translateValueName(createTranslator('en'), 'customValue')).toBe('customValue');
    expect(translatePort(createTranslator('en'), 'output', 'probabilities')).toBe('Probabilities');
    expect(translatePort(createTranslator('zh-CN'), 'output', 'probabilities')).toBe('概率');
  });
  it('translates component-specific ports before shared labels and preserves unknown ports', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      for (const locale of ['en', 'zh-CN'] as const) {
        const t = createTranslator(locale);
        expect(translatePort(t, 'one_hot', 'k')).toBe(locale === 'en' ? 'classes k' : '类别数 k');
        for (const component of ['attention', 'attn_scores', 'multi_head'])
          expect(translatePort(t, component, 'k')).toBe(locale === 'en' ? 'key k' : '键 k');
        expect(translatePort(t, 'reshape', 'd0')).toBe(t('ports.d0'));
        expect(translatePort(t, 'custom', 'x')).toBe(t('ports.x'));
        expect(translatePort(t, 'custom', 'custom_port')).toBe('custom_port');
        expect(translatePort(t, 'custom', 'toString')).toBe('toString');
      }
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
  it('falls back to English for a missing component-specific port translation', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ports.one_hot.k'];
    delete dictionary['ports.one_hot.k'];
    try {
      expect(translatePort(createTranslator('zh-CN'), 'one_hot', 'k')).toBe('classes k');
    } finally {
      dictionary['ports.one_hot.k'] = original;
      spy.mockRestore();
    }
  });
  it('localizes automatic-module wiring diagnostics', () => {
    expect(
      translateDiagnostic('en', { key: 'generated_input_connected', port: 'w', params: {} }),
    ).toContain('managed by this module');
    expect(
      translateDiagnostic('zh-CN', { key: 'generated_input_connected', port: 'w', params: {} }),
    ).toContain('自动管理');
  });
  it('renders shape-variant diagnostics with localized scalar input shapes', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const text = translateDiagnostic(locale, {
        key: 'levels.shape_variant',
        params: { shapes: { a: [3, 2], t: [] } },
      });
      expect(text).toContain('a [3×2]');
      expect(text).toContain(`t ${locales[locale]['ui.scalar']}`);
      expect(text).not.toContain('[]');
      expect(text).not.toContain('{shapes}');
      expect(text).not.toBe('levels.shape_variant');
    }
  });
  it('renders empty shape parameters as scalar without relabeling an empty axis list', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const mismatch = translateDiagnostic(locale, {
        key: 'input_shape_mismatch',
        params: { actual: [], expected: [3] },
      });
      expect(mismatch).toContain(locales[locale]['ui.scalar']);
      expect(mismatch).not.toContain('[]');
      const axes = translateDiagnostic(locale, {
        key: 'invalid_axes',
        params: { actual: [0], rank: 0, allowed: [] },
      });
      expect(axes).toContain(locales[locale]['ui.none']);
      expect(axes).not.toContain(locales[locale]['ui.scalar']);
      expect(axes).not.toContain('[]');
    }
  });
  it('labels shape variants as tests and describes scalar dot-product output plainly', () => {
    expect(locales.en['ui.shapeGroup']).toBe('Test shape');
    expect(locales['zh-CN']['ui.shapeGroup']).toBe('测试形状');
    for (const locale of ['en', 'zh-CN'] as const) {
      expect(locales[locale]['levels.axes.description']).toContain(
        locales[locale]['ui.shapeGroup'],
      );
      expect(locales[locale]['levels.2-1.description']).not.toContain('$y$ []');
      expect(locales[locale]['levels.2-1.hint2']).not.toContain('shape []');
    }
    expect(locales.en['levels.2-1.description']).toContain('output is scalar');
    expect(locales['zh-CN']['levels.2-1.description']).toContain('输出是标量');
  });
  it('distinguishes broadcast element-wise multiplication from matrix multiplication', () => {
    const english = [
      locales.en['levels.2-2.description'],
      locales.en['levels.2-2.hint1'],
      locales.en['levels.2-2.hint2'],
    ].join(' ');
    const chinese = [
      locales['zh-CN']['levels.2-2.description'],
      locales['zh-CN']['levels.2-2.hint1'],
      locales['zh-CN']['levels.2-2.hint2'],
    ].join(' ');
    expect(english).toContain('$x ⊙ s + b$');
    expect(english).toMatch(/element-wise multiplication.*not matrix multiplication/);
    expect(chinese).toContain('$x ⊙ s + b$');
    expect(chinese).toMatch(/逐元素乘法.*不是矩阵乘法/);
  });
  it('teaches shape-safe mean with Dimension size in both locales', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const description = locales[locale]['levels.2-3.description'];
      const hints = `${locales[locale]['levels.2-3.hint1']} ${locales[locale]['levels.2-3.hint2']}`;
      expect(description).toMatch(/Dimension size|维度大小/);
      expect(description).not.toMatch(/Passing unlocks|过关.*得到/);
      expect(hints).toMatch(/axis −1|轴 −1/);
      expect(hints).toMatch(/Dimension size|维度大小/);
      expect(hints).not.toContain('x × 0 + 1');
    }
  });
  it('introduces gradient and backpropagation before separate loss and descent games', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const surface = locales[locale]['levels.1-10.description'];
      const backprop = locales[locale]['levels.1-11.description'];
      const backpropHint1 = locales[locale]['levels.1-11.hint1'];
      const backpropHint2 = locales[locale]['levels.1-11.hint2'];
      const loss = locales[locale]['levels.loss-intro.description'];
      const descent = locales[locale]['levels.3-1.description'];
      expect(surface).toContain('$∇f=(∂f/∂a,∂f/∂b)$');
      expect(surface).not.toMatch(/负梯度|negative gradient|gradient descent|梯度下降|∇L|η/);
      expect(backprop).toContain('$$\nz=w·x+b\n$$');
      expect(backprop).toContain('$$\nf=σ(z)\n$$');
      expect(backprop).toContain('$$\ng=\\frac{∂f}{∂z}\n$$');
      expect(backprop).toMatch(/上游梯度|upstream gradient/);
      expect(backprop).toMatch(
        /只(?:反传|处理) \$z=w·x\+b\$|backpropagate only through \$z=w·x\+b\$/,
      );
      expect(backprop).toMatch(/不(?:需要重新计算|用再算) Sigmoid|without recomputing Sigmoid/);
      expect(backprop).toMatch(/没有(?:提供|给).*b|not provided/);
      expect(backprop).not.toMatch(/∂z\/∂w=x|∂z\/∂x=w|∂z\/∂b=1/);
      expect(backprop).not.toMatch(/∂f\/∂w=g·x|∂f\/∂x=g·w|∂f\/∂b=g/);
      expect(backpropHint1).toContain('$∂f/∂q=(∂f/∂z)·(∂z/∂q)$');
      expect(backpropHint1).not.toMatch(/∂z\/∂w=x|∂f\/∂w=g·x/);
      expect(backpropHint2).toContain('$∂z/∂w=x$');
      expect(backpropHint2).toContain('$∂z/∂x=w$');
      expect(backpropHint2).toContain('$∂z/∂b=1$');
      expect(backpropHint2).toContain('$∂f/∂w=g·x$');
      expect(backpropHint2).toContain('$∂f/∂x=g·w$');
      expect(backpropHint2).toContain('$∂f/∂b$');
      for (const text of [backprop, backpropHint1, backpropHint2]) {
        expect(text).not.toMatch(/损失|loss|训练|train|梯度下降|gradient descent/);
        expect(text).not.toMatch(/\bLinear\b|线性层/);
      }
      expect(loss).toMatch(/模型根据输入算出的答案叫预测|model's answer computed from its inputs/i);
      expect(loss).toMatch(/直接给出|are given/);
      expect(loss).toContain('$L=(p−t)²$');
      expect(loss).toContain('$p=t$');
      expect(loss).toMatch(/拖动|Drag/);
      expect(loss).not.toMatch(/∂L\/∂w|w′|η/);
      expect(descent).toMatch(/上一关|previous level/i);
      expect(descent).toMatch(/最简单的预测器|simplest predictor/);
      expect(descent).toContain('1-11');
      expect(descent).toMatch(/上游梯度|upstream gradient/i);
      expect(descent).toContain('$g=∂L/∂p$');
      expect(descent).toContain('$∂L/∂w=(∂L/∂p)·x$');
      expect(descent).toContain('$w′=w−0.1·∂L/∂w$');
      expect(descent).toContain('$η=0.1$');
      expect(descent).not.toContain('[]');
      expect(locales[locale]['plot.gradientDirectionHelp']).toBeTruthy();
      for (const key of [
        'lossFlow',
        'descentFlow',
        'lossControl',
        'descentControl',
        'curve',
        'current',
        'minimum',
        'gradientDirection',
        'targetStep',
        'yourLoss',
        'yourUpdate',
        'help',
      ])
        expect((locales[locale] as Record<string, unknown>)[`plot.loss.${key}`]).toBeTruthy();
    }
  });
  it('introduces a multi-feature predictor before its weight-gradient lesson', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const neuron = locales[locale]['levels.neuron.description'];
      const gradient = locales[locale]['levels.3-2.description'];
      const hints = `${locales[locale]['levels.3-2.hint1']} ${locales[locale]['levels.3-2.hint2']}`;
      expect(neuron).toMatch(/显示 3-2|display 3-2/i);
      expect(neuron).toMatch(/神经元|neuron/i);
      expect(neuron).toMatch(/预测|prediction/i);
      expect(neuron).toContain('$$\np=w_{1}x_{1}+w_{2}x_{2}+b\n$$');
      expect(neuron).toContain('$p=w·x+b$');
      expect(neuron).toContain('$Xw+b$');
      expect(neuron).toContain('$XW+b$');
      expect(neuron).toMatch(/矩阵.*每一行.*点积|matrix.*Each row.*dot product/is);
      expect(gradient).toMatch(
        /前面的搭建关造过一个神经元|previous construction level built one neuron/i,
      );
      expect(gradient).toMatch(/偏置固定为 0|bias is fixed at 0/i);
      expect(gradient).toContain('$p=w·x$');
      expect(gradient).toContain('1-10');
      expect(gradient).toContain('1-11');
      expect(gradient).toContain('$g=∂L/∂p$');
      expect(gradient).toContain('$∂L/∂w=g·x$');
      expect(`${gradient} ${hints}`).toMatch(/上游梯度|upstream gradient/i);
      expect(hints).toContain('$∂L/∂q=(∂L/∂p)·(∂p/∂q)$');
      expect(hints).toMatch(/without −0\.1|不做 −0\.1/);
      expect(`${gradient} ${hints}`).not.toContain('$2(p−t)$');
      expect(`${gradient} ${hints}`).not.toMatch(/∇|∂L\/∂w[₁₂₃ᵢ]/);
    }
  });
  it('explains how to build k and why one-hot matrix multiplication performs a lookup', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const description = locales[locale]['levels.4-2.description'];
      const hints = `${locales[locale]['levels.4-2.hint1']} ${locales[locale]['levels.4-2.hint2']}`;
      expect(description).toMatch(/独热编码.*idx 和 k|One-hot.*idx and k/is);
      expect(`${description} ${hints}`).toMatch(/table.*第 0 轴|axis 0.*table/is);
      expect(`${description} ${hints}`).toMatch(/序列.*k=\[0|Range.*k=\[0/is);
      expect(description).toContain('0A+0B+1C+0D+0E=C');
      expect(description).toMatch(/0.*其他行.*1.*保留 C|zeroes.*every other row.*1 keeps C/is);
      expect(description).toMatch(/idx=\[2,0\].*C.*A/is);
      expect(description).toContain('\\operatorname{one\\_hot}(idx,k)');
      expect(description).toContain('\\times');
      expect(description).toMatch(/不能把 5 或 k 写死|hard-coding 5 or k/i);
    }
  });
  it('contrasts fixed-window character prediction with recurrent memory', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const oneCharacter = locales[locale]['levels.4-3.description'];
      const twoCharacters = locales[locale]['levels.4-5.description'];
      const recurrent = locales[locale]['levels.time-rnn.description'];
      expect(oneCharacter).toContain('$P(x_{t+1}\\mid x_t)$');
      expect(oneCharacter).toMatch(/不是有记忆|无记忆|memoryless|not yet.*memory/is);
      expect(oneCharacter).toMatch(/lit.*cat/is);
      expect(oneCharacter).toMatch(/RNN.*隐藏状态|RNN.*hidden state/is);
      expect(twoCharacters).toContain('$P(x_{t+1}\\mid x_{t-1},x_t)$');
      expect(twoCharacters).toMatch(/顺序.*重要|Order matters/i);
      expect(twoCharacters).toMatch(/上下文长度.*固定为 2|context length.*fixed at 2/is);
      expect(twoCharacters).toMatch(/没有隐藏状态|no hidden state/i);
      expect(recurrent).toMatch(/固定后缀|fixed suffix/i);
      expect(recurrent).toMatch(/隐藏状态.*前面|hidden state.*earlier/is);
      expect(recurrent).toMatch(/复用同一组权重|Reusing the same weights/i);
    }
  });
  it('keeps the fixed-window observation before building RNN', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const t = createTranslator(locale);
      const note = t('levels.time-window-limit.demonstrationNote');
      const explanation = t('levels.time-window-limit.explanation');
      const next = t('levels.time-window-limit.next');
      expect(t('levels.time-window-limit.description')).toContain('4-6');
      expect(t('levels.time-window-limit.description')).not.toContain('RNN');
      expect(t('levels.time-window-limit.afterTraining')).toContain('160');
      expect(t('levels.time-window-limit.afterMoreTraining')).toMatch(/160.*320/);
      expect(note).toMatch(/隔离实验|Isolated experiment/i);
      expect(note).toMatch(/不读取你的训练权重|not your saved weights/i);
      expect(note).toMatch(/同一模型和优化器|same model and optimizer/i);
      expect(t('levels.time-window-limit.sameOutput')).toContain('th');
      expect(explanation).toMatch(/最多答对一个|at most one answer/i);
      expect(explanation).toMatch(/随机采样.*碰巧|Sampling may differ by chance/i);
      expect(explanation).toMatch(/不是对整体准确率|not an overall accuracy/i);
      expect(next).toContain('RNN');
      expect(next).toMatch(/记忆并非无限|not unlimited memory/i);
      expect(t('levels.time-window-limit.continue')).toMatch(/搭建 RNN|Build the RNN/);
    }
  });
  it('describes the post-build experiment as retraining both player circuits, not old weights or a fallback', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const t = createTranslator(locale);
      const text = (key: string) => t(`levels.time-player-comparison.${key}`);
      const comparison = text('comparisonNote');
      expect(text('freshWeights')).toMatch(
        /使用你的电路重新训练，非沿用旧权重|Retrain using your circuits, not your old weights/i,
      );
      expect(text('freshWeights')).not.toMatch(/预制|prebuilt/i);
      expect(text('freshWeights')).toMatch(
        /8 维字符嵌入和 24→27 输出层|8-wide character embedding, a 24→27 output layer/i,
      );
      expect(comparison).toContain('1,495');
      expect(comparison).toMatch(
        /全语料行内每个目标位置 ≥ 2|every within-line target position ≥ 2 in the full corpus/i,
      );
      expect(comparison).toMatch(/含末尾换行|including the final newline/i);
      expect(comparison).toMatch(
        /更新数、Adam、学习率和种子相同|Same update count, Adam, learning rate and seed/i,
      );
      expect(comparison).toMatch(/参数量和算力不同|parameter counts and compute differ/i);
      expect(comparison).toMatch(
        /训练语料内的拟合实验，不是泛化评测|training-corpus fitting, not a generalization test/i,
      );
      expect(text('resetNote')).toMatch(
        /同一组权重，只变输入，不额外训练|Same weights, only the input changes; no extra training/i,
      );
      expect(text('conclusion')).toMatch(/不保证答对|does not guarantee a correct answer/i);
      expect(text('failed')).toMatch(/对照实验未能完成|comparison experiment could not finish/i);
      expect(text('failed')).not.toMatch(/预制|无效|修复|prebuilt|invalid|must be fixed/i);
      expect(text('continue')).toMatch(/下一关|Next level/);
      expect(text('back')).toMatch(/返回编辑|Back to editing/);
    }
  });
  it('keeps both experiment namespaces in parity and retires reference-RNN and manual controls', () => {
    for (const prefix of ['levels.time-window-limit.', 'levels.time-player-comparison.']) {
      const keys = (dictionary: object) =>
        Object.keys(dictionary)
          .filter((key) => key.startsWith(prefix))
          .sort();
      expect(keys(locales.en)).toEqual(keys(locales['zh-CN']));
      for (const locale of ['en', 'zh-CN'] as const) {
        const t = createTranslator(locale);
        for (const key of keys(locales[locale]))
          expect(t(key)).not.toMatch(/两个连线|two wires|take_last_two|50\s*%/i);
        expect(t(prefix + 'case', { n: 2 })).toBe(locale === 'en' ? 'Sample 2' : '样本 2');
        expect(t(prefix + 'trainingProgress', { model: 'RNN', step: 170, total: 320 })).toContain(
          '170 / 320',
        );
      }
    }
    for (const locale of ['en', 'zh-CN'] as const) {
      const dictionary = locales[locale];
      const t = createTranslator(locale);
      for (const key of [
        'predictSpace',
        'predictE',
        'sharedPrediction',
        'tryPrediction',
        'runRecurrent',
        'resetHistory',
        'comparisonNote',
      ]) {
        expect(Object.hasOwn(dictionary, 'levels.time-window-limit.' + key)).toBe(false);
      }
      expect(Object.hasOwn(dictionary, 'ports.take_last_two.history')).toBe(false);
      expect(Object.hasOwn(dictionary, 'components.take_last_two')).toBe(false);
      expect(t('levels.time-player-comparison.matchedSnapshot', { step: 320 })).toContain('320');
      expect(
        t('levels.time-player-comparison.modelMetrics', {
          model: 'RNN',
          parameters: 1234,
          samples: 1495,
        }),
      ).toMatch(/RNN.*1,234.*1,495/);
    }
  });
  it('explains probability sampling from intervals through the actual 4-4 circuit', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const description = locales[locale]['levels.4-4.description'];
      const hints = `${locales[locale]['levels.4-4.hint1']} ${locales[locale]['levels.4-4.hint2']}`;
      expect(description).toMatch(/概率最大的字|highest probability/i);
      expect(description).toMatch(/区间.*宽.*概率|interval.*width.*probability/is);
      expect(description).toContain('p=[0.2,0.5,0.3]');
      expect(description).toContain('c=[0.2,0.7,1]');
      expect(description).toMatch(/u=0\.6.*输出 1|u=0\.6.*outputs 1/is);
      expect(`${description} ${hints}`).toMatch(/插入维度|Insert dimension/i);
      expect(`${description} ${hints}`).toMatch(/逐元素相乘|element-wise multiplication/i);
      expect(`${description} ${hints}`).toMatch(/轴 0.*求和|Sum along axis 0/is);
      expect(hints).toMatch(/u≥c/);
      expect(hints).toContain('[1,0,0]');
      expect(hints).toMatch(/最后一轴.*求和|Sum along the last axis/i);
    }
  });
  it('connects MSE visually to squared loss and chapter 2 Mean', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const description = locales[locale]['levels.3-3.description'];
      const visual = locales[locale]['visual.mse.help'];
      expect(description).toMatch(/显示 3-1|Display 3-1/i);
      expect(description).toMatch(/显示 2-4|Display 2-4/i);
      expect(`${description} ${visual}`).toMatch(/平方损失|squared loss/i);
      expect(`${description} ${visual}`).toMatch(/平均值|Mean/);
      for (const key of [
        'title',
        'help',
        'prediction',
        'target',
        'errors',
        'squares',
        'output',
        'mean',
      ])
        expect((locales[locale] as Record<string, unknown>)[`visual.mse.${key}`]).toBeTruthy();
    }
  });
  it('explains cross entropy with a localized row-selectable batch trace', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const description = locales[locale]['levels.3-5.description'];
      const visual = locales[locale]['visual.cross_entropy.help'];
      expect(`${description} ${visual}`).toMatch(/独热|one-hot/i);
      expect(`${description} ${visual}`).toMatch(/每行|per-row|row losses/i);
      expect(`${description} ${visual}`).toMatch(/整批|batch/i);
      for (const key of [
        'title',
        'help',
        'probabilities',
        'logs',
        'target',
        'selected',
        'rowLoss',
        'batchMean',
      ])
        expect(
          (locales[locale] as Record<string, unknown>)[`visual.cross_entropy.${key}`],
        ).toBeTruthy();
    }
  });
  it('teaches momentum as two recurrent state outputs', () => {
    const level = gameLevels.find(
      (candidate) => candidate.kind === 'forward' && candidate.id === '3-7',
    )!;
    if (level.kind !== 'forward') throw new Error('Momentum must be a circuit lesson');
    expect(level.interface.outputs).toEqual({ updatedWeight: [], updatedMomentum: [] });
    for (const locale of ['en', 'zh-CN'] as const) {
      const text = [
        locales[locale]['levels.3-7.description'],
        locales[locale]['levels.3-7.hint1'],
        locales[locale]['levels.3-7.hint2'],
      ].join(' ');
      expect(text).toContain('$w′$');
      expect(text).toContain('$m′$');
      expect(text).toMatch(/两个输出|Both .* required outputs|outputs are/i);
      expect(locales[locale]['values.updatedMomentum']).toBeTruthy();
    }
  });
  it('keeps the Adam comparison focused on gameplay', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const text = [
        locales[locale]['levels.3-8.description'],
        locales[locale]['levels.3-8.hint1'],
        locales[locale]['levels.3-8.hint2'],
      ].join(' ');
      expect(text).toContain('3-10');
      expect(locales[locale]['loop.update.sgd']).toMatch(
        /SGD.*(?:普通梯度下降|plain gradient descent)/is,
      );
      expect(text).toMatch(/SGD.*(?:普通梯度下降|plain gradient descent)/is);
      expect(text).toMatch(/不同参数|different parameters/i);
      expect(text).toContain('×0.05');
      expect(text).toContain('×20');
      expect(text).toMatch(/SGD.*(?:重置|reset).*Adam/is);
      expect(text).toMatch(/损失和准确率|loss and accuracy/i);
      expect(text).not.toMatch(
        /一阶矩|二阶矩|平方梯度|first moment|second moment|squared gradients?|√v|bias corrections?/i,
      );
      expect(locales[locale]['loop.update.adam']).toMatch(/3-11.*3-12/);
    }
  });
  it('teaches configuring learned modules instead of manually wiring Parameters', () => {
    const levelIds = ['3-6', '3-8', '3-9', '4-3', '4-5', '5-5', '6-4'] as const;
    for (const locale of ['en', 'zh-CN'] as const) {
      const hints = Object.fromEntries(
        levelIds.map((id) => [
          id,
          `${locales[locale][`levels.${id}.hint1`]} ${locales[locale][`levels.${id}.hint2`]}`,
        ]),
      );
      for (const [id, text] of Object.entries(hints)) {
        expect(text, `${locale}: ${id}`).toMatch(locale === 'en' ? /[Cc]onfigur/ : /配置/);
        expect(text, `${locale}: ${id}`).not.toMatch(
          locale === 'en'
            ? /\bParameter\b|parameter table|Gather|Matrix multiply plus a bias/
            : /「参数」|参数表|索引选取|「矩阵乘法」/,
        );
      }

      expect(hints['3-6']).toMatch(/Linear|线性层/);
      expect(hints['3-6']).toMatch(
        /2.*input features.*2.*output features|输入特征数和输出特征数都配置为 2/,
      );
      expect(hints['3-8']).toMatch(/Linear|线性层/);
      expect(hints['3-8']).toContain('$W$ [2×2]');
      expect(hints['3-9']).toMatch(/2.*16.*16.*2/s);
      expect(hints['4-3']).toMatch(/Embedding|嵌入/);
      expect(hints['4-3']).toContain('[V×V]');
      expect(hints['4-5']).toMatch(/Embedding|嵌入/);
      expect(hints['4-5']).toMatch(/Linear|线性层/);
      expect(hints['5-5']).toMatch(/Token \+ position embedding|词元 \+ 位置嵌入/);
      expect(hints['5-5']).toMatch(/27.*8.*12.*12→12.*12→27/s);
      expect(hints['6-4']).toMatch(/Token \+ position embedding|词元 \+ 位置嵌入/);
      expect(hints['6-4']).toMatch(
        /two separate Transformer block|两个相互独立的「Transformer 块」/,
      );
      expect(hints['6-4']).toMatch(/27.*16.*24.*24.*48.*24→27/s);
    }
  });
  it('derives native language names for every registered locale', () => {
    expect(Object.keys(localeNames)).toEqual(Object.keys(locales));
    for (const locale of Object.keys(locales) as (keyof typeof locales)[]) {
      expect(localeNames[locale]).toBe(
        new Intl.DisplayNames([locale], { type: 'language' }).of(locale) ?? locale,
      );
    }
  });
  it('matches browser language preferences in order', () => {
    expect(matchLocale(['fr', 'zh-TW', 'en'])).toBe('zh-CN');
    expect(matchLocale(['en-GB', 'zh-CN'])).toBe('en');
    expect(matchLocale(['ZH-cn'])).toBe('zh-CN');
    expect(matchLocale(['fr'])).toBe('en');
  });
  it('uses Intl plural rules, number formatting, and explicit locale factories', () => {
    const t = createTranslator('en');
    expect(t('ui.nodeCount', { count: 1 })).toBe('1 node');
    expect(t('ui.nodeCount', { count: 1234 })).toBe('1,234 nodes');
    expect(translate('zh-CN', 'ui.nodeCount', { count: 1 })).toBe('1 个节点');
    expect(formatNumber('en', 0.25, { style: 'percent' })).toBe('25%');
    expect(t('ui.step')).toBe('Step {count}');
  });
  it('falls back to English and warns in development without throwing', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ui.save'];
    delete dictionary['ui.save'];
    try {
      expect(translate('zh-CN', 'ui.save')).toBe('Save');
      expect(translate('en', 'not.a.key')).toBe('not.a.key');
      expect(translate('en', 'toString')).toBe('toString');
      expect(spy).toHaveBeenCalled();
    } finally {
      dictionary['ui.save'] = original;
      spy.mockRestore();
    }
  });
  it('uses English plural rules when falling back to an English message', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ui.nodeCount'];
    delete dictionary['ui.nodeCount'];
    try {
      expect(translate('zh-CN', 'ui.nodeCount', { count: 1 })).toBe('1 node');
      expect(translate('zh-CN', 'ui.nodeCount', { count: 2 })).toBe('2 nodes');
    } finally {
      dictionary['ui.nodeCount'] = original;
      spy.mockRestore();
    }
  });
  it('covers every registered component, port and example level with both languages', () => {
    expect(Object.keys(locales.en).sort()).toEqual(Object.keys(locales['zh-CN']).sort());
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      for (const component of createDefaultRegistry().list()) {
        expect(keys.has(`components.${component.name}`)).toBe(true);
        for (const port of [...component.inputs, ...component.outputs])
          expect(keys.has(`ports.${port}`)).toBe(true);
      }
      for (const level of [...exampleLevels, ...gameLevels]) {
        expect(keys.has(level.titleKey)).toBe(true);
        expect(keys.has(level.descriptionKey)).toBe(true);
        for (const key of level.hintsKeys ?? []) expect(keys.has(key)).toBe(true);
        for (const name of level.unlockComponents ?? [])
          expect(keys.has(`components.${name}`)).toBe(true);
        if (level.kind === 'observation') continue;
        for (const name of level.available) expect(keys.has(`components.${name}`)).toBe(true);
        for (const output of Object.keys(
          level.interface.outputs ?? { [level.interface.output]: level.interface.outputShape },
        )) {
          if (output === 'y' || /[′∂]/.test(output)) continue;
          expect(keys.has(`values.${output}`), `${locale}: values.${output}`).toBe(true);
        }
      }
    }
  });
  it('uses the requested Transformer lesson title in both languages', () => {
    expect(locales.en['levels.6-3.title']).toBe('Attention Is All You Need');
    expect(locales['zh-CN']['levels.6-3.title']).toBe('注意力是你需要的一切');
  });
  it('covers recurrent settings and component-specific bias ports in both languages', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      expect(keys.has('ui.input_features')).toBe(true);
      for (const component of ['rnn_cell', 'rnn_window'])
        expect(keys.has(`ports.${component}.b`)).toBe(true);
    }
  });
  it('covers the temporal benchmark evidence in both languages', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      for (const key of [
        'benchmark.title',
        'benchmark.seed',
        'benchmark.model',
        'benchmark.parameters',
        'benchmark.delay',
        'benchmark.method',
        'benchmark.attentionProbe',
        'benchmark.position',
        'benchmark.attentionProbeNote',
        'benchmark.model.fixed-window',
        'benchmark.model.rnn',
        'benchmark.model.rnn-attention',
      ])
        expect(keys.has(key), `${locale}: ${key}`).toBe(true);
      expect(locales[locale]['benchmark.method']).toMatch(/CPU/);
      expect(locales[locale]['benchmark.method']).toContain('[0.5,1.5)');
      expect(locales[locale]['benchmark.method']).toContain('[−0.2,0.2)');
    }
  });
  it('covers GPT weight ports in both languages', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      for (const port of ['wq', 'wk', 'wv', 'wo', 'w1', 'b1', 'w2', 'b2'])
        expect(keys.has(`ports.${port}`)).toBe(true);
    }
  });
  it('localizes component parameters in both languages and preserves custom names', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      for (const [key, parameter] of [
        ['component_unavailable', 'component'],
        ['missing_component', 'requiredComponent'],
      ] as const) {
        expect(translateDiagnostic(locale, { key, params: { [parameter]: 'matmul' } })).toBe(
          translate(locale, `errors.${key}`, {
            [parameter]: locales[locale]['components.matmul'],
          }),
        );
        expect(
          translateDiagnostic(locale, { key, params: { [parameter]: 'custom_part' } }),
        ).toContain('custom_part');
      }
    }
  });
  it('logs original internal diagnostic codes and params once across locales', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      for (const key of ['engine.tensor.disposed', 'disposed_plan', 'unrecognized_test_error']) {
        const error = { key, params: { message: 'private backend detail', component: 'matmul' } };
        for (const locale of ['en', 'zh-CN'] as const) {
          const text = translateDiagnostic(locale, error);
          expect(text).toBe(locales[locale]['errors.unknown']);
          expect(text).not.toContain(key);
          expect(text).not.toContain(error.params.message);
        }
        expect(spy).toHaveBeenCalledWith('[i18n] Diagnostic:', key, error.params, error);
      }
      expect(spy).toHaveBeenCalledTimes(3);
      translateDiagnostic('en', { key: 'missing_input', params: { node: 'a', port: 'x' } });
      expect(spy).toHaveBeenCalledTimes(3);
    } finally {
      spy.mockRestore();
    }
  });
  it('maps raw and namespaced diagnostics without exposing raw exceptions', () => {
    expect(diagnosticKey('missing_input')).toBe('errors.missing_input');
    expect(diagnosticKey('training.nonfinite')).toBe('training.nonfinite');
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      for (const locale of ['en', 'zh-CN'] as const) {
        expect(translateDiagnostic(locale, { message: 'private raw exception' })).toBe(
          locales[locale]['errors.unknown'],
        );
        expect(
          translateDiagnostic(locale, { key: 'component_error', params: { message: 'raw' } }),
        ).not.toContain('raw');
      }
    } finally {
      spy.mockRestore();
    }
  });
});
