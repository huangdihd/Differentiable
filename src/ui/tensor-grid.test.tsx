import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import TensorGrid from './TensorGrid';

it('labels matrix rows and columns without changing their values', () => {
  const html = renderToStaticMarkup(
    <TensorGrid
      label="window"
      shape={[2, 2]}
      values={[0.25, -0.75, 0.25, -0.75]}
      rows={['target=1', 'target=-1']}
      cols={['T−2', 'T−1']}
      format={String}
    />,
  );

  expect(html.match(/scope="row"/g)).toHaveLength(2);
  expect(html).toContain('T−2');
  expect(html).toContain('T−1');
  expect(html.match(/>0.25<\/td>/g)).toHaveLength(2);
  expect(html.match(/>-0.75<\/td>/g)).toHaveLength(2);
  expect(html).toContain('target=1');
  expect(html).toContain('target=-1');
});
