import type { Locale } from '../i18n';

/**
 * Beginner-friendly reading for concepts beyond middle-school mathematics/computing. A concept is
 * linked only where it first becomes necessary, so later levels are not crowded with repeats.
 */
export interface LearningResource {
  titleKey: string;
  descriptionKey: string;
  url: Readonly<Record<Locale, string>>;
}

const d2l = (zhCN: string, en: string): LearningResource['url'] => ({ 'zh-CN': zhCN, en });

export const levelLearningResources: Readonly<Record<string, readonly LearningResource[]>> = {
  '0-5': [
    {
      titleKey: 'learning.activations.title',
      descriptionKey: 'learning.activations.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html',
        'https://en.d2l.ai/chapter_multilayer-perceptrons/mlp.html',
      ),
    },
  ],
  '1-1': [
    {
      titleKey: 'learning.calculus.title',
      descriptionKey: 'learning.calculus.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_preliminaries/calculus.html',
        'https://en.d2l.ai/chapter_preliminaries/calculus.html',
      ),
    },
  ],
  '1-8': [
    {
      titleKey: 'learning.multivariable.title',
      descriptionKey: 'learning.multivariable.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_preliminaries/calculus.html#id6',
        'https://en.d2l.ai/chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.html',
      ),
    },
  ],
  '1-10': [
    {
      titleKey: 'learning.backprop.title',
      descriptionKey: 'learning.backprop.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_multilayer-perceptrons/backprop.html',
        'https://en.d2l.ai/chapter_multilayer-perceptrons/backprop.html',
      ),
    },
  ],
  axes: [
    {
      titleKey: 'learning.linearAlgebra.title',
      descriptionKey: 'learning.linearAlgebra.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html',
        'https://en.d2l.ai/chapter_preliminaries/linear-algebra.html',
      ),
    },
  ],
  '2-7': [
    {
      titleKey: 'learning.softmax.title',
      descriptionKey: 'learning.softmax.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html',
        'https://en.d2l.ai/chapter_linear-classification/softmax-regression.html',
      ),
    },
  ],
  '3-1': [
    {
      titleKey: 'learning.gradientDescent.title',
      descriptionKey: 'learning.gradientDescent.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_optimization/gd.html',
        'https://en.d2l.ai/chapter_optimization/gd.html',
      ),
    },
  ],
  '4-1': [
    {
      titleKey: 'learning.oneHot.title',
      descriptionKey: 'learning.oneHot.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html',
        'https://en.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html',
      ),
    },
  ],
  '4-2': [
    {
      titleKey: 'learning.embeddings.title',
      descriptionKey: 'learning.embeddings.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html',
        'https://en.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html',
      ),
    },
  ],
  '5-1': [
    {
      titleKey: 'learning.attention.title',
      descriptionKey: 'learning.attention.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html',
        'https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html',
      ),
    },
  ],
  '6-3': [
    {
      titleKey: 'learning.transformer.title',
      descriptionKey: 'learning.transformer.description',
      url: d2l(
        'https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html',
        'https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html',
      ),
    },
  ],
};
