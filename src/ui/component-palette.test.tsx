import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { ComponentPalette, groupPaletteEntries, type PaletteEntry } from './ComponentPalette';

const t = (key: string) => key;

describe('component palette', () => {
  it('groups families in a stable order and keeps inverse operations adjacent', () => {
    const entries: PaletteEntry[] = [
      { name: 'relu' },
      { name: 'sqrt' },
      { name: 'sub' },
      { name: 'exp' },
      { name: 'add' },
      { name: 'div' },
      { name: 'mul' },
      { name: 'log' },
      { name: 'square' },
      { name: 'dsigmoid' },
      { name: 'rnn_window' },
      { name: 'param' },
      { name: 'rnn_cell' },
      { name: 'temporal_attention' },
      { name: 'attention' },
      { name: 'custom' },
    ];
    const groups = groupPaletteEntries(entries);
    expect(groups.map((group) => group.category)).toEqual([
      'op',
      'act',
      'grad',
      'learn',
      'attn',
      'other',
    ]);
    expect(groups[0].entries.map((entry) => entry.name)).toEqual([
      'add',
      'sub',
      'mul',
      'div',
      'exp',
      'log',
      'square',
      'sqrt',
    ]);
    expect(
      groups.find((group) => group.category === 'learn')?.entries.map((entry) => entry.name),
    ).toEqual(['param', 'rnn_cell', 'rnn_window']);
    expect(
      groups.find((group) => group.category === 'attn')?.entries.map((entry) => entry.name),
    ).toEqual(['temporal_attention', 'attention']);
  });

  it('server-renders independently open groups, counts and entry states', () => {
    const html = renderToStaticMarkup(
      <ComponentPalette
        entries={[
          { name: 'add' },
          { name: 'sub', locked: true },
          { name: 'const' },
          { name: 'sigmoid', player: true, isNew: true },
        ]}
        t={t}
        disabled={false}
        onPress={vi.fn()}
        onActivate={vi.fn()}
      />,
    );
    expect(html.match(/<details/g)).toHaveLength(2);
    expect(html.match(/ open=""/g)).toHaveLength(2);
    expect(html).toContain('ui.componentCategory.op');
    expect(html).toContain('ui.componentCategory.act');
    expect(html).toContain('<span class="palette-count">3</span>');
    expect(html).toContain('data-component="add"');
    expect(html).toContain('data-pair-start="true"');
    expect(html).toContain('class="palette-item is-locked"');
    expect(html).toContain('components.sub · ui.producedHere');
    expect(html).toContain('class="palette-item is-player is-new"');
    expect(html).toContain('<span class="new-badge palette-new">ui.new</span>');
    expect(html).toMatch(/data-component="const" data-category="op"/);
  });
});
