import { useState, type PointerEvent } from 'react';
import { SYMBOLS, categoryOf, type Category } from './graph-geometry';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;
type PaletteCategory = Category | 'other';

export interface PaletteEntry {
  name: string;
  player?: boolean;
  locked?: boolean;
  isNew?: boolean;
}

const CATEGORY_ORDER: readonly PaletteCategory[] = [
  'op',
  'act',
  'tensor',
  'grad',
  'learn',
  'attn',
  'other',
];

/** Related operations stay adjacent: add/sub, mul/div, exp/log, square/square-root, etc. */
const COMPONENT_ORDER = [
  'add',
  'sub',
  'mul',
  'div',
  'exp',
  'log',
  'square',
  'sqrt',
  'neg',
  'recip',
  'max',
  'ge',
  'const',
  'sigmoid',
  'dsigmoid',
  'tanh',
  'dtanh',
  'relu',
  'softmax',
  'sum',
  'mean',
  'reduce_max',
  'reshape',
  'size',
  'unsqueeze',
  'transpose',
  'slice',
  'stack',
  'concat',
  'split',
  'gather',
  'scatter_add',
  'arange',
  'dot',
  'outer',
  'matmul',
  'one_hot',
  'embedding',
  'layernorm',
  'param',
  'linear',
  'mse',
  'cross_entropy',
  'rnn_cell',
  'rnn_window',
  'sample',
  'temporal_attention',
  'embed_pos',
  'attn_scores',
  'causal_mask',
  'attention',
  'multi_head',
  'feed_forward',
  'transformer_block',
] as const;
const COMPONENT_RANK = new Map<string, number>(COMPONENT_ORDER.map((name, index) => [name, index]));
const PAIRS: Readonly<Record<string, string>> = {
  add: 'sub',
  mul: 'div',
  exp: 'log',
  square: 'sqrt',
};

/** Groups and orders the library into the same colour-coded families used on the canvas. */
export function groupPaletteEntries(
  entries: readonly PaletteEntry[],
): { category: PaletteCategory; entries: PaletteEntry[] }[] {
  const groups = new Map<PaletteCategory, PaletteEntry[]>();
  for (const entry of entries) {
    const category = categoryOf(entry.name) ?? 'other';
    groups.set(category, [...(groups.get(category) ?? []), entry]);
  }
  for (const grouped of groups.values())
    grouped.sort(
      (a, b) =>
        (COMPONENT_RANK.get(a.name) ?? Number.MAX_SAFE_INTEGER) -
        (COMPONENT_RANK.get(b.name) ?? Number.MAX_SAFE_INTEGER),
    );
  return CATEGORY_ORDER.flatMap((category) => {
    const grouped = groups.get(category);
    return grouped?.length ? [{ category, entries: grouped }] : [];
  });
}

/** A component library whose colour families can be folded independently as it grows. */
export function ComponentPalette({
  entries,
  t,
  disabled,
  onPress,
  onActivate,
}: {
  entries: readonly PaletteEntry[];
  t: T;
  disabled: boolean;
  onPress: (component: string, event: PointerEvent) => void;
  onActivate: (component: string) => void;
}) {
  return (
    <div className="palette-groups">
      {groupPaletteEntries(entries).map((group) => (
        <PaletteGroup
          key={group.category}
          category={group.category}
          entries={group.entries}
          t={t}
          disabled={disabled}
          onPress={onPress}
          onActivate={onActivate}
        />
      ))}
    </div>
  );
}

function PaletteGroup({
  category,
  entries,
  t,
  disabled,
  onPress,
  onActivate,
}: {
  category: PaletteCategory;
  entries: readonly PaletteEntry[];
  t: T;
  disabled: boolean;
  onPress: (component: string, event: PointerEvent) => void;
  onActivate: (component: string) => void;
}) {
  const [open, setOpen] = useState(true);
  return (
    <details
      className="palette-group"
      data-category={category === 'other' ? undefined : category}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span>{t(`ui.componentCategory.${category}`)}</span>
        <span className="palette-count">{entries.length}</span>
      </summary>
      <div className="palette">
        {entries.map((entry, index) => (
          <PaletteItem
            key={entry.name}
            entry={entry}
            pairStart={
              index % 2 === 0 &&
              PAIRS[entry.name] !== undefined &&
              PAIRS[entry.name] === entries[index + 1]?.name
            }
            t={t}
            disabled={disabled}
            onPress={onPress}
            onActivate={onActivate}
          />
        ))}
      </div>
    </details>
  );
}

function PaletteItem({
  entry,
  pairStart,
  t,
  disabled,
  onPress,
  onActivate,
}: {
  entry: PaletteEntry;
  pairStart: boolean;
  t: T;
  disabled: boolean;
  onPress: (component: string, event: PointerEvent) => void;
  onActivate: (component: string) => void;
}) {
  const content = (
    <>
      <span className={`sym ${(SYMBOLS[entry.name] ?? '').length > 6 ? 'is-long' : ''}`}>
        {SYMBOLS[entry.name] ?? '?'}
      </span>
      <span className="name">
        {t(`components.${entry.name}`)}
        {entry.locked ? ` · ${t('ui.producedHere')}` : ''}
        {entry.isNew && <span className="new-badge palette-new">{t('ui.new')}</span>}
      </span>
    </>
  );
  if (entry.locked)
    return (
      <div
        className="palette-item is-locked"
        data-component={entry.name}
        data-category={categoryOf(entry.name)}
        data-pair-start={pairStart || undefined}
        title={t('ui.producedHere')}
      >
        {content}
      </div>
    );
  return (
    <button
      type="button"
      className={['palette-item', entry.player && 'is-player', entry.isNew && 'is-new']
        .filter(Boolean)
        .join(' ')}
      data-component={entry.name}
      data-category={categoryOf(entry.name)}
      data-pair-start={pairStart || undefined}
      disabled={disabled}
      onPointerDown={(event) => onPress(entry.name, event)}
      onClick={(event) => event.detail === 0 && onActivate(entry.name)}
    >
      {content}
    </button>
  );
}
