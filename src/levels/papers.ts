/** The original papers behind a level's idea, shown as further reading. Only levels whose
 * concept has a clear primary source are listed; titles stay in their original language. */
export interface Paper {
  title: string;
  authors: string;
  year: number;
  url: string;
}

const P = {
  mcculloch: {
    title: 'A logical calculus of the ideas immanent in nervous activity',
    authors: 'McCulloch, Pitts',
    year: 1943,
    url: 'https://doi.org/10.1007/BF02478259',
  },
  shannon48: {
    title: 'A Mathematical Theory of Communication',
    authors: 'Shannon',
    year: 1948,
    url: 'https://doi.org/10.1002/j.1538-7305.1948.tb01338.x',
  },
  shannon51: {
    title: 'Prediction and Entropy of Printed English',
    authors: 'Shannon',
    year: 1951,
    url: 'https://doi.org/10.1002/j.1538-7305.1951.tb01366.x',
  },
  rosenblatt: {
    title:
      'The perceptron: a probabilistic model for information storage and organization in the brain',
    authors: 'Rosenblatt',
    year: 1958,
    url: 'https://doi.org/10.1037/h0042519',
  },
  polyak: {
    title: 'Some methods of speeding up the convergence of iteration methods',
    authors: 'Polyak',
    year: 1964,
    url: 'https://doi.org/10.1016/0041-5553(64)90137-5',
  },
  linnainmaa: {
    title: 'Taylor expansion of the accumulated rounding error',
    authors: 'Linnainmaa',
    year: 1976,
    url: 'https://doi.org/10.1007/BF01931367',
  },
  rumelhart: {
    title: 'Learning representations by back-propagating errors',
    authors: 'Rumelhart, Hinton, Williams',
    year: 1986,
    url: 'https://doi.org/10.1038/323533a0',
  },
  bengio: {
    title: 'A Neural Probabilistic Language Model',
    authors: 'Bengio, Ducharme, Vincent, Jauvin',
    year: 2003,
    url: 'https://www.jmlr.org/papers/v3/bengio03a.html',
  },
  glorot: {
    title: 'Deep Sparse Rectifier Neural Networks',
    authors: 'Glorot, Bordes, Bengio',
    year: 2011,
    url: 'https://proceedings.mlr.press/v15/glorot11a.html',
  },
  adam: {
    title: 'Adam: A Method for Stochastic Optimization',
    authors: 'Kingma, Ba',
    year: 2014,
    url: 'https://arxiv.org/abs/1412.6980',
  },
  bahdanau: {
    title: 'Neural Machine Translation by Jointly Learning to Align and Translate',
    authors: 'Bahdanau, Cho, Bengio',
    year: 2014,
    url: 'https://arxiv.org/abs/1409.0473',
  },
  resnet: {
    title: 'Deep Residual Learning for Image Recognition',
    authors: 'He, Zhang, Ren, Sun',
    year: 2015,
    url: 'https://arxiv.org/abs/1512.03385',
  },
  layernorm: {
    title: 'Layer Normalization',
    authors: 'Ba, Kiros, Hinton',
    year: 2016,
    url: 'https://arxiv.org/abs/1607.06450',
  },
  transformer: {
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    url: 'https://arxiv.org/abs/1706.03762',
  },
  gpt: {
    title: 'Improving Language Understanding by Generative Pre-Training',
    authors: 'Radford, Narasimhan, Salimans, Sutskever',
    year: 2018,
    url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf',
  },
  gpt2: {
    title: 'Language Models are Unsupervised Multitask Learners',
    authors: 'Radford et al.',
    year: 2019,
    url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
  },
} satisfies Record<string, Paper>;

export const levelPapers: Readonly<Record<string, readonly Paper[]>> = {
  '0-7': [P.glorot],
  '1-10': [P.linnainmaa, P.rumelhart],
  neuron: [P.mcculloch],
  '2-8': [P.layernorm],
  '3-5': [P.shannon48],
  '3-6': [P.rosenblatt],
  '3-7': [P.polyak],
  '3-8': [P.adam],
  '3-9': [P.rumelhart],
  xor: [P.rumelhart],
  '4-2': [P.bengio],
  '4-3': [P.shannon51, P.bengio],
  '5-1': [P.bahdanau],
  '5-2': [P.transformer],
  '5-3': [P.transformer],
  '5-4': [P.transformer],
  '6-1': [P.transformer],
  '6-2': [P.transformer],
  '6-3': [P.transformer, P.resnet],
  '6-4': [P.gpt, P.gpt2],
};
