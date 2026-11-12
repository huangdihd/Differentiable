// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Sandbox mode: a hub of local saves, and the editor with three tabs beside the canvas —
 * training, data and saves. Nothing leaves the browser; files are read with FileReader and
 * written as downloads.
 */
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react';
import type { GraphJSON } from '../engine/graph';
import type { Progress, TrainLevel } from '../engine/levels';
import type { Locale } from '../i18n';
import { formatNumber } from '../i18n';
import type { WorkerClient } from '../worker/client';
import { WorkerClientError } from '../worker/client';
import {
  DatasetImportError,
  columnInfo,
  guessMapping,
  importedShapes,
  mapCsv,
  mappingFromRoles,
  parseCsv,
  parseJsonDataset,
  type ColumnRole,
  type CsvTable,
  type ImportedDataset,
} from '../sandbox/dataset-import';
import { SANDBOX_EXAMPLES, builtinTask, type SandboxRows } from '../sandbox/datasets';
import { editorProgress, fromEditorProgress } from '../sandbox/editor-progress';
import { buildSandboxLevel, datasetKey } from '../sandbox/level';
import { datasetRows, datasetStarter, datasetTask, rowsPlot, summarize } from '../sandbox/preview';
import {
  MAX_SLOTS,
  createSlot,
  duplicateSlot,
  exportWeights,
  parseWeights,
  removeSlot,
  sandboxLevelId,
  sandboxSlots,
  uniqueName,
  updateSlot,
  type SandboxDataset,
  type SandboxSlot,
} from '../sandbox/slots';
import { starterGraph } from '../sandbox/starter';
import DataPlot from './DataPlot';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export type SandboxTab = 'train' | 'data' | 'saves';

/** What the sandbox hands the Editor: the panels around the canvas and a starter model. */
export interface SandboxEditorSlot {
  header: ReactNode;
  /** Prototype layouts: back, name and save state for the toolbar row above the canvas. */
  barHead: ReactNode;
  /** Short label of the slot's task (regression / classification) for the data button. */
  taskLabel: string;
  dataPanel: ReactNode;
  savesPanel: ReactNode;
  tab: SandboxTab;
  onTab: (tab: SandboxTab) => void;
  view: 'fit' | 'boundary' | 'none';
  starter: () => GraphJSON;
}

const newId = () => {
  try {
    return crypto.randomUUID();
  } catch {
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
};

export const readFile = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });

export function download(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

const errorText = (t: T, error: unknown) =>
  error instanceof DatasetImportError
    ? t(error.key, error.params)
    : error instanceof WorkerClientError
      ? t(error.detail.key, error.detail.params as Record<string, string | number>)
      : error instanceof Error && error.message.startsWith('sandbox.')
        ? t(error.message)
        : t('sandbox.import.error.read');

export const datasetLabel = (t: T, dataset: SandboxDataset) =>
  dataset.kind === 'builtin' ? t(`datasets.${dataset.id}.name`) : dataset.name;

const nodeCount = (slot: SandboxSlot) =>
  slot.graph?.nodes.filter((n) => n.component !== 'input' && n.component !== 'output').length ?? 0;

function relativeTime(locale: Locale, then: number, now = Date.now()) {
  const seconds = Math.round((then - now) / 1000);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  const abs = Math.abs(seconds);
  if (abs < 45) return rtf.format(0, 'second');
  if (abs < 3600) return rtf.format(Math.round(seconds / 60), 'minute');
  if (abs < 86400) return rtf.format(Math.round(seconds / 3600), 'hour');
  if (abs < 86400 * 30) return rtf.format(Math.round(seconds / 86400), 'day');
  return new Date(then).toLocaleDateString(locale);
}

const clockTime = (locale: Locale, at: number) =>
  new Date(at).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });

/** Rows of a dataset for previews; imported rows are already at hand. */
function useRows(dataset: SandboxDataset): SandboxRows | undefined {
  const key = datasetKey(dataset);
  const [rows, setRows] = useState<{ key: string; rows: SandboxRows }>();
  const current = useRef(dataset);
  current.current = dataset;
  useEffect(() => {
    let alive = true;
    void datasetRows(current.current).then((r) => alive && setRows({ key, rows: r }));
    return () => {
      alive = false;
    };
  }, [key]);
  return rows?.key === key ? rows.rows : undefined;
}

export function SandboxScreen({
  t,
  locale,
  progress,
  onProgress,
  ready,
  client,
  onExit,
  renderEditor,
}: {
  t: T;
  locale: Locale;
  progress: Progress;
  onProgress: (update: (current: Progress) => Progress) => void;
  ready: boolean;
  client: React.RefObject<WorkerClient | null>;
  onExit: () => void;
  renderEditor: (props: {
    key: string;
    level: TrainLevel;
    progress: Progress;
    onProgress: (p: Progress) => void;
    sandbox: SandboxEditorSlot;
  }) => ReactNode;
}) {
  const [open, setOpen] = useState<string>();
  const [tab, setTab] = useState<SandboxTab>('train');
  const slots = sandboxSlots(progress);
  const slot = slots.find((s) => s.id === open);
  const create = (dataset: SandboxDataset) => {
    const id = newId();
    onProgress((current) =>
      createSlot(current, {
        id,
        name: uniqueName(
          datasetLabel(t, dataset),
          sandboxSlots(current).map((s) => s.name),
        ),
        now: Date.now(),
        dataset,
      }),
    );
    setTab('train');
    setOpen(id);
  };
  const duplicate = (id: string) => {
    const copy = newId();
    onProgress((current) => {
      const source = sandboxSlots(current).find((s) => s.id === id);
      if (!source) return current;
      return duplicateSlot(current, id, {
        id: copy,
        name: uniqueName(
          t('sandbox.slot.copyName', { name: source.name }),
          sandboxSlots(current).map((s) => s.name),
        ),
        now: Date.now(),
      });
    });
    return copy;
  };
  if (!slot)
    return (
      <SandboxHub
        t={t}
        locale={locale}
        slots={slots}
        ready={ready}
        onExit={onExit}
        onOpen={(id) => {
          setTab('train');
          setOpen(id);
        }}
        onCreate={create}
        onRename={(id, name) => onProgress((p) => updateSlot(p, id, { name }, Date.now()))}
        onDuplicate={duplicate}
        onDelete={(id) => onProgress((p) => removeSlot(p, id))}
      />
    );
  return (
    <Workspace
      t={t}
      locale={locale}
      slot={slot}
      full={slots.length >= MAX_SLOTS}
      progress={progress}
      onProgress={onProgress}
      client={client}
      tab={tab}
      onTab={setTab}
      onBack={() => setOpen(undefined)}
      onDuplicate={() => duplicate(slot.id)}
      onDelete={() => {
        setOpen(undefined);
        onProgress((p) => removeSlot(p, slot.id));
      }}
      renderEditor={renderEditor}
    />
  );
}

/* ---------------------------------------------------------------- hub */

function SandboxHub({
  t,
  locale,
  slots,
  ready,
  onExit,
  onOpen,
  onCreate,
  onRename,
  onDuplicate,
  onDelete,
}: {
  t: T;
  locale: Locale;
  slots: SandboxSlot[];
  ready: boolean;
  onExit: () => void;
  onOpen: (id: string) => void;
  onCreate: (dataset: SandboxDataset) => void;
  onRename: (id: string, name: string) => void;
  onDuplicate: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [picking, setPicking] = useState(false);
  const [importing, setImporting] = useState(false);
  const full = slots.length >= MAX_SLOTS;
  const sorted = [...slots].sort((a, b) => b.updatedAt - a.updatedAt);
  const pickTitle = useId();
  return (
    <main className="home blueprint-bg sb-hub">
      <div className="home-inner">
        <button className="sb-back" onClick={onExit}>
          ← {t('sandbox.backToRoad')}
        </button>
        <header className="sb-hub-head">
          <div>
            <span className="road-n">∞</span>
            <h1>{t('sandbox.title')}</h1>
            <p className="muted">{t('sandbox.hub.lead')}</p>
          </div>
          {slots.length > 0 && (
            <button
              className="btn btn-primary sb-new"
              disabled={!ready || full}
              title={full ? t('sandbox.slot.full', { max: MAX_SLOTS }) : undefined}
              onClick={() => setPicking(true)}
            >
              + {t('sandbox.slot.new')}
            </button>
          )}
        </header>
        {slots.length === 0 ? (
          <section className="sb-start" aria-labelledby={pickTitle}>
            <h2 id={pickTitle} className="sb-section-title">
              {t('sandbox.hub.startTitle')}
            </h2>
            <p className="muted">{t('sandbox.hub.startLead')}</p>
            <DatasetPicker
              t={t}
              disabled={!ready}
              onPick={onCreate}
              onImport={() => setImporting(true)}
            />
            <p className="hint">{t('sandbox.dataset.localOnly')}</p>
          </section>
        ) : (
          <section aria-label={t('sandbox.hub.slots')}>
            <div className="sb-section-row">
              <h2 className="sb-section-title">{t('sandbox.hub.slots')}</h2>
              <span className="label">
                {t('sandbox.hub.slotCount', { n: slots.length, max: MAX_SLOTS })}
              </span>
            </div>
            <ul className="sb-slots">
              {sorted.map((slot) => (
                <SlotCard
                  key={slot.id}
                  t={t}
                  locale={locale}
                  slot={slot}
                  ready={ready}
                  full={full}
                  onOpen={() => onOpen(slot.id)}
                  onRename={(name) => onRename(slot.id, name)}
                  onDuplicate={() => onDuplicate(slot.id)}
                  onDelete={() => onDelete(slot.id)}
                />
              ))}
            </ul>
            <p className="hint">{t('sandbox.hub.storageNote')}</p>
          </section>
        )}
      </div>
      {picking && (
        <Dialog t={t} title={t('sandbox.new.title')} onClose={() => setPicking(false)} wide>
          <p className="muted sb-dialog-lead">{t('sandbox.new.lead')}</p>
          <DatasetPicker
            t={t}
            onPick={(dataset) => {
              setPicking(false);
              onCreate(dataset);
            }}
            onImport={() => {
              setPicking(false);
              setImporting(true);
            }}
          />
        </Dialog>
      )}
      {importing && (
        <ImportDialog
          t={t}
          locale={locale}
          onCancel={() => setImporting(false)}
          onConfirm={(data) => {
            setImporting(false);
            onCreate({ kind: 'imported', ...data });
          }}
        />
      )}
    </main>
  );
}

function SlotCard({
  t,
  locale,
  slot,
  ready,
  full,
  onOpen,
  onRename,
  onDuplicate,
  onDelete,
}: {
  t: T;
  locale: Locale;
  slot: SandboxSlot;
  ready: boolean;
  full: boolean;
  onOpen: () => void;
  onRename: (name: string) => void;
  onDuplicate: () => void;
  onDelete: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(slot.name);
  const [confirming, setConfirming] = useState(false);
  const nodes = nodeCount(slot);
  const name = slot.name.trim() || t('sandbox.slot.untitled');
  return (
    <li className={`sb-slot${confirming ? ' is-confirming' : ''}`}>
      <button
        className="sb-slot-open"
        disabled={!ready || renaming}
        onClick={onOpen}
        aria-label={t('sandbox.slot.openNamed', { name })}
      >
        <DatasetThumb t={t} dataset={slot.dataset} />
        <span className="sb-slot-text">
          {renaming ? null : <span className="sb-slot-name">{name}</span>}
          {datasetLabel(t, slot.dataset) !== name && (
            <span className="sb-slot-meta">{datasetLabel(t, slot.dataset)}</span>
          )}
          <span className="sb-slot-meta">
            {nodes ? t('sandbox.slot.nodes', { count: nodes }) : t('sandbox.slot.empty')} ·{' '}
            {relativeTime(locale, slot.updatedAt)}
          </span>
        </span>
      </button>
      {renaming && (
        <form
          className="sb-rename"
          onSubmit={(e) => {
            e.preventDefault();
            onRename(draft.trim() || slot.name);
            setRenaming(false);
          }}
        >
          <input
            className="field"
            autoFocus
            value={draft}
            maxLength={40}
            aria-label={t('sandbox.slot.name')}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setRenaming(false);
            }}
          />
          <button className="btn btn-primary btn-small">{t('sandbox.slot.save')}</button>
          <button type="button" className="btn btn-small" onClick={() => setRenaming(false)}>
            {t('sandbox.cancel')}
          </button>
        </form>
      )}
      {confirming ? (
        <div className="sb-confirm" role="alertdialog" aria-label={t('sandbox.slot.delete')}>
          <span>{t('sandbox.slot.deleteConfirm', { name })}</span>
          <span className="btn-row">
            <button className="btn btn-small btn-danger is-armed" onClick={onDelete} autoFocus>
              {t('sandbox.slot.delete')}
            </button>
            <button className="btn btn-small" onClick={() => setConfirming(false)}>
              {t('sandbox.cancel')}
            </button>
          </span>
        </div>
      ) : (
        !renaming && (
          <div className="sb-slot-actions">
            <button className="btn btn-small btn-primary" disabled={!ready} onClick={onOpen}>
              {t('sandbox.slot.open')}
            </button>
            <button
              className="btn btn-small"
              onClick={() => {
                setDraft(slot.name);
                setRenaming(true);
              }}
            >
              {t('sandbox.slot.rename')}
            </button>
            <button
              className="btn btn-small"
              disabled={full}
              title={full ? t('sandbox.slot.full', { max: MAX_SLOTS }) : undefined}
              onClick={onDuplicate}
            >
              {t('sandbox.slot.duplicate')}
            </button>
            <button className="btn btn-small btn-danger" onClick={() => setConfirming(true)}>
              {t('sandbox.slot.delete')}
            </button>
          </div>
        )
      )}
    </li>
  );
}

/* ---------------------------------------------------------------- data pickers */

function DatasetPicker({
  t,
  current,
  compact,
  disabled,
  onPick,
  onImport,
}: {
  t: T;
  current?: SandboxDataset;
  compact?: boolean;
  disabled?: boolean;
  onPick: (dataset: SandboxDataset) => void;
  onImport: () => void;
}) {
  return (
    <div className={`sb-picker${compact ? ' is-compact' : ''}`}>
      {SANDBOX_EXAMPLES.map((id) => {
        const selected = current?.kind === 'builtin' && current.id === id;
        return (
          <button
            key={id}
            className={`sb-tile${selected ? ' is-current' : ''}`}
            aria-pressed={current ? selected : undefined}
            disabled={disabled || selected}
            onClick={() => onPick({ kind: 'builtin', id })}
          >
            <DatasetThumb t={t} dataset={{ kind: 'builtin', id }} />
            <span className="sb-tile-name">{t(`datasets.${id}.name`)}</span>
            <span className="sb-tile-meta">
              {t(`sandbox.task.${builtinTask(id)}`)}
              {selected ? ` · ${t('sandbox.dataset.inUse')}` : ''}
            </span>
          </button>
        );
      })}
      <button className="sb-tile sb-tile-import" disabled={disabled} onClick={onImport}>
        <span className="sb-thumb sb-thumb-import" aria-hidden="true">
          <span>CSV</span>
          <span>JSON</span>
        </span>
        <span className="sb-tile-name">{t('sandbox.import.tile')}</span>
        <span className="sb-tile-meta">{t('sandbox.import.tileHint')}</span>
      </button>
    </div>
  );
}

const TW = 120,
  TH = 72,
  TP = 7;

/** A small drawing of the data: dots on a curve, or coloured classes in the plane. */
function DatasetThumb({ t, dataset }: { t: T; dataset: SandboxDataset }) {
  const rows = useRows(dataset);
  const task = datasetTask(dataset);
  const plot = rows ? rowsPlot(rows, task) : undefined;
  if (!rows) return <span className="sb-thumb" aria-hidden="true" />;
  if (!plot)
    return (
      <span className="sb-thumb sb-thumb-table" aria-hidden="true">
        {t('sandbox.dataset.tableShape', {
          rows: rows.x.length,
          columns: (rows.x[0]?.length ?? 0) + (rows.t[0]?.length ?? 0),
        })}
      </span>
    );
  const xs = plot.kind === 'regression' ? plot.x : plot.points.map((p) => p[0]);
  const ys = plot.kind === 'regression' ? plot.t : plot.points.map((p) => p[1]);
  const span = (v: number[]) => {
    const lo = Math.min(...v),
      hi = Math.max(...v);
    return [lo, hi - lo || 1] as const;
  };
  const [x0, xw] = span(xs),
    [y0, yh] = span(ys);
  const sx = (v: number) => TP + ((v - x0) / xw) * (TW - 2 * TP);
  const sy = (v: number) => TH - TP - ((v - y0) / yh) * (TH - 2 * TP);
  return (
    <svg className="sb-thumb" viewBox={`0 0 ${TW} ${TH}`} aria-hidden="true">
      {xs.map((x, i) => (
        <circle
          key={i}
          className={`plot-point${plot.kind === 'classes' ? ` is-class-${plot.label[i] % 4}` : ''}`}
          cx={sx(x)}
          cy={sy(ys[i])}
          r={2}
        />
      ))}
    </svg>
  );
}

/* ---------------------------------------------------------------- workspace */

function Workspace({
  t,
  locale,
  slot,
  full,
  progress,
  onProgress,
  client,
  tab,
  onTab,
  onBack,
  onDuplicate,
  onDelete,
  renderEditor,
}: {
  t: T;
  locale: Locale;
  slot: SandboxSlot;
  full: boolean;
  progress: Progress;
  onProgress: (update: (current: Progress) => Progress) => void;
  client: React.RefObject<WorkerClient | null>;
  tab: SandboxTab;
  onTab: (tab: SandboxTab) => void;
  onBack: () => void;
  onDuplicate: () => string;
  onDelete: () => void;
  renderEditor: Parameters<typeof SandboxScreen>[0]['renderEditor'];
}) {
  const levelId = sandboxLevelId(slot.id);
  const dataKey = datasetKey(slot.dataset);
  // Any change to the weights file (import, clear, a different file) resyncs the worker.
  const weightsKey = useMemo(() => {
    if (!slot.weights) return 0;
    const text = JSON.stringify(slot.weights);
    let h = 0;
    for (let i = 0; i < text.length; i++) h = (Math.imul(h, 31) + text.charCodeAt(i)) | 0;
    return `${text.length}:${h >>> 0}`;
  }, [slot.weights]);
  const level = useMemo<TrainLevel | Error>(() => {
    try {
      return buildSandboxLevel(slot.id, slot.dataset);
    } catch (error) {
      return error as Error;
    }
  }, [slot.id, dataKey]);
  // The worker must know this slot before the Editor's first compile reaches it.
  const syncKey = `${slot.id}|${dataKey}|${weightsKey}`;
  const [synced, setSynced] = useState<string>();
  const [syncError, setSyncError] = useState<string>();
  const [importing, setImporting] = useState(false);
  const current = useRef(slot);
  current.current = slot;
  useEffect(() => {
    let alive = true;
    setSynced(undefined);
    setSyncError(undefined);
    const s = current.current;
    client.current
      ?.setSandbox({
        slotId: s.id,
        dataset: s.dataset,
        ...(s.weights ? { weights: s.weights } : {}),
      })
      .then(
        () => alive && setSynced(syncKey),
        (error) => alive && setSyncError(errorText(t, error)),
      );
    return () => {
      alive = false;
    };
  }, [syncKey]);
  const change = (update: Partial<Omit<SandboxSlot, 'id' | 'createdAt'>>) =>
    onProgress((p) => updateSlot(p, slot.id, update, Date.now()));
  const task = datasetTask(slot.dataset);

  const header = (
    <div className="section sb-head">
      <button className="sb-back" onClick={onBack}>
        ← {t('sandbox.allSlots')}
      </button>
      <label className="sb-name">
        <span className="level-id">∞ {t('sandbox.title')}</span>
        <input
          className="sb-name-input"
          value={slot.name}
          maxLength={40}
          placeholder={t('sandbox.slot.untitled')}
          aria-label={t('sandbox.slot.name')}
          onChange={(e) => change({ name: e.target.value })}
        />
      </label>
      <p className="sb-saved">
        <span aria-hidden="true">✓</span>{' '}
        {t('sandbox.autosaved', { time: clockTime(locale, slot.updatedAt) })}
      </p>
      <p className="sb-head-data">
        <span className="tag">{t(`sandbox.task.${task}`)}</span> {datasetLabel(t, slot.dataset)}
      </p>
    </div>
  );

  const barHead = (
    <div className="sb-bar-head">
      <button className="btn btn-small sb-bar-back" onClick={onBack} title={t('sandbox.allSlots')}>
        ← {t('sandbox.bar.back')}
      </button>
      <span className="sb-bar-mode" aria-hidden="true">
        ∞
      </span>
      <input
        className="sb-name-input sb-bar-name"
        value={slot.name}
        maxLength={40}
        placeholder={t('sandbox.slot.untitled')}
        aria-label={t('sandbox.slot.name')}
        title={t('sandbox.bar.rename')}
        onChange={(e) => change({ name: e.target.value })}
      />
      <span className="sb-saved sb-bar-saved">
        <span aria-hidden="true">✓</span>{' '}
        {t('sandbox.autosaved', { time: clockTime(locale, slot.updatedAt) })}
      </span>
    </div>
  );

  const dataPanel = (
    <DataPanel
      t={t}
      slot={slot}
      onPick={(dataset) => change({ dataset })}
      onImport={() => setImporting(true)}
    />
  );
  const savesPanel = (
    <SavesPanel
      t={t}
      locale={locale}
      slot={slot}
      full={full}
      client={client}
      levelId={levelId}
      onWeights={(weights) => change({ weights })}
      onDuplicate={onDuplicate}
      onDelete={onDelete}
    />
  );

  const dialogs = importing && (
    <ImportDialog
      t={t}
      locale={locale}
      onCancel={() => setImporting(false)}
      onConfirm={(data) => {
        setImporting(false);
        change({ dataset: { kind: 'imported', ...data } });
      }}
    />
  );

  if (level instanceof Error || syncError)
    return (
      <main className="home blueprint-bg">
        <div className="home-inner sb-problem" role="alert">
          <h2>{t('sandbox.problem.title')}</h2>
          <p className="muted">
            {syncError ?? errorText(t, level)} {t('sandbox.problem.body')}
          </p>
          <div className="btn-row">
            <button
              className="btn btn-primary"
              onClick={() => change({ dataset: { kind: 'builtin', id: 'noisy-line' } })}
            >
              {t('sandbox.problem.reset')}
            </button>
            <button className="btn" onClick={onBack}>
              ← {t('sandbox.allSlots')}
            </button>
          </div>
        </div>
      </main>
    );
  if (synced !== syncKey)
    return (
      <main className="home blueprint-bg">
        <div className="home-inner sb-loading" role="status">
          {t('sandbox.loading')}
        </div>
      </main>
    );
  const shapes = { x: level.interface.inputs.x, t: level.interface.inputs.t };
  const view =
    task === 'regression' && shapes.x.length === 1
      ? 'fit'
      : task === 'classification' && shapes.x.length === 2 && shapes.x[1] === 2
        ? 'boundary'
        : 'none';
  return (
    <>
      {renderEditor({
        key: syncKey,
        level,
        progress: editorProgress(progress, slot, levelId),
        onProgress: (next) => onProgress(() => fromEditorProgress(next, slot.id, Date.now())),
        sandbox: {
          header,
          barHead,
          taskLabel: t(`sandbox.task.${task}`),
          dataPanel,
          savesPanel,
          tab,
          onTab,
          view,
          starter: () => starterGraph(shapes, task, datasetStarter(slot.dataset)),
        },
      })}
      {dialogs}
    </>
  );
}

function DataPanel({
  t,
  slot,
  onPick,
  onImport,
}: {
  t: T;
  slot: SandboxSlot;
  onPick: (dataset: SandboxDataset) => void;
  onImport: () => void;
}) {
  const rows = useRows(slot.dataset);
  const task = datasetTask(slot.dataset);
  const plot = rows ? rowsPlot(rows, task) : undefined;
  const summary = rows ? summarize(rows, task) : undefined;
  const shapes = rows ? importedShapes(rows) : undefined;
  return (
    <div className="sb-data">
      <section className="sb-card">
        <header className="sb-card-head">
          <h3>{datasetLabel(t, slot.dataset)}</h3>
          <span className="tag">{t(`sandbox.task.${task}`)}</span>
        </header>
        <p className="sb-card-text">
          {slot.dataset.kind === 'builtin'
            ? t(`datasets.${slot.dataset.id}.source`)
            : t('sandbox.dataset.importedSource')}
        </p>
        {plot && <DataPlot data={plot} label={datasetLabel(t, slot.dataset)} />}
        {summary && shapes && (
          <dl className="kv sb-kv">
            <dt>{t('sandbox.dataset.rows')}</dt>
            <dd>{summary.rows}</dd>
            <dt>{t('sandbox.dataset.input')}</dt>
            <dd>x [{shapes.x.join('×')}]</dd>
            <dt>{t('sandbox.dataset.target')}</dt>
            <dd>
              t [{shapes.t.join('×')}]
              {task === 'classification'
                ? ` · ${t('sandbox.dataset.classes', { n: summary.outputs, count: summary.outputs })}`
                : ''}
            </dd>
          </dl>
        )}
      </section>
      <section className="sb-block">
        <h3 className="sb-block-title">{t('sandbox.dataset.switch')}</h3>
        <DatasetPicker t={t} compact current={slot.dataset} onPick={onPick} onImport={onImport} />
        <p className="hint">{t('sandbox.dataset.switchHint')}</p>
      </section>
    </div>
  );
}

function SavesPanel({
  t,
  locale,
  slot,
  full,
  client,
  levelId,
  onWeights,
  onDuplicate,
  onDelete,
}: {
  t: T;
  locale: Locale;
  slot: SandboxSlot;
  full: boolean;
  client: React.RefObject<WorkerClient | null>;
  levelId: string;
  onWeights: (weights: SandboxSlot['weights']) => void;
  onDuplicate: () => string;
  onDelete: () => void;
}) {
  const [message, setMessage] = useState<{ text: string; bad?: boolean }>();
  const [confirming, setConfirming] = useState(false);
  const weightsInput = useRef<HTMLInputElement>(null);
  const imported = slot.weights ? Object.keys(slot.weights.parameters).length : 0;
  const when = (at: number) =>
    new Date(at).toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' });
  return (
    <div className="sb-saves">
      <section className="sb-card">
        <header className="sb-card-head">
          <h3>{t('sandbox.weights.title')}</h3>
        </header>
        <p className={`sb-weights-state${imported ? ' is-file' : ''}`}>
          {imported
            ? t('sandbox.weights.fromFile', { n: imported })
            : t('sandbox.weights.fromRandom')}
        </p>
        <div className="sb-actions">
          <button
            className="btn"
            onClick={() => {
              setMessage(undefined);
              client.current
                ?.exportWeights(levelId, slot.graph ?? { version: 1, nodes: [], edges: [] })
                .then(({ parameters }) => {
                  download(
                    `${slot.name.trim() || 'sandbox'}.weights.json`,
                    exportWeights(parameters),
                  );
                  setMessage({ text: t('sandbox.weights.exported') });
                })
                .catch((error) => setMessage({ text: errorText(t, error), bad: true }));
            }}
          >
            ⇩ {t('sandbox.weights.export')}
          </button>
          <button className="btn" onClick={() => weightsInput.current?.click()}>
            ⇧ {t('sandbox.weights.import')}
          </button>
          {imported > 0 && (
            <button
              className="btn"
              onClick={() => {
                onWeights(undefined);
                setMessage({ text: t('sandbox.weights.cleared') });
              }}
            >
              ↺ {t('sandbox.weights.clear')}
            </button>
          )}
        </div>
        <input
          ref={weightsInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (!file) return;
            try {
              const weights = parseWeights(await readFile(file));
              onWeights(weights);
              setMessage({
                text: t('sandbox.weights.imported', { n: Object.keys(weights.parameters).length }),
              });
            } catch (error) {
              setMessage({ text: errorText(t, error), bad: true });
            }
          }}
        />
        <p className="hint">{t('sandbox.weights.help')}</p>
        {message && (
          <p className={`sb-message${message.bad ? ' is-bad' : ''}`} role="status">
            {message.text}
          </p>
        )}
      </section>
      <section className="sb-card">
        <header className="sb-card-head">
          <h3>{t('sandbox.slot.title')}</h3>
        </header>
        <dl className="kv sb-kv">
          <dt>{t('sandbox.slot.created')}</dt>
          <dd>{when(slot.createdAt)}</dd>
          <dt>{t('sandbox.slot.saved')}</dt>
          <dd>{when(slot.updatedAt)}</dd>
          <dt>{t('sandbox.slot.parts')}</dt>
          <dd>{formatNumber(locale, nodeCount(slot))}</dd>
        </dl>
        <div className="sb-actions">
          <button
            className="btn"
            disabled={full}
            title={full ? t('sandbox.slot.full', { max: MAX_SLOTS }) : undefined}
            onClick={() => {
              onDuplicate();
              setMessage({ text: t('sandbox.slot.duplicated') });
            }}
          >
            ⧉ {t('sandbox.slot.duplicate')}
          </button>
          {!confirming && (
            <button className="btn btn-danger" onClick={() => setConfirming(true)}>
              {t('sandbox.slot.delete')}
            </button>
          )}
        </div>
        {confirming && (
          <div className="sb-confirm" role="alertdialog" aria-label={t('sandbox.slot.delete')}>
            <span>
              {t('sandbox.slot.deleteConfirm', {
                name: slot.name.trim() || t('sandbox.slot.untitled'),
              })}
            </span>
            <span className="btn-row">
              <button className="btn btn-small btn-danger is-armed" onClick={onDelete} autoFocus>
                {t('sandbox.slot.delete')}
              </button>
              <button className="btn btn-small" onClick={() => setConfirming(false)}>
                {t('sandbox.cancel')}
              </button>
            </span>
          </div>
        )}
        <p className="hint">{t('sandbox.hub.storageNote')}</p>
      </section>
    </div>
  );
}

/* ---------------------------------------------------------------- dialogs */

export function Dialog({
  t,
  title,
  onClose,
  wide,
  medium,
  children,
  footer,
}: {
  t: T;
  title: string;
  onClose: () => void;
  wide?: boolean;
  medium?: boolean;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const titleId = useId();
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose]);
  return (
    <div className="modal-back" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <section
        className={`modal sb-dialog${wide ? ' is-wide' : medium ? ' is-medium' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <header className="sb-dialog-head">
          <h2 id={titleId}>{title}</h2>
          <button
            className="icon-btn sb-dialog-close"
            aria-label={t('sandbox.close')}
            onClick={onClose}
          >
            ×
          </button>
        </header>
        <div className="sb-dialog-body">{children}</div>
        {footer && <div className="btn-row modal-actions">{footer}</div>}
      </section>
    </div>
  );
}

const PREVIEW_ROWS = 6;
const ROLES: readonly ColumnRole[] = ['x', 't', 'skip'];

/**
 * Pick a file, then one screen: every column with an x / t / skip switch over its first rows,
 * and a live preview of what the model will see. JSON files skip the column step.
 */
export function ImportDialog({
  t,
  locale,
  onCancel,
  onConfirm,
}: {
  t: T;
  locale: Locale;
  onCancel: () => void;
  onConfirm: (data: ImportedDataset) => void;
}) {
  const [fileName, setFileName] = useState('');
  const [readError, setReadError] = useState<string>();
  const [table, setTable] = useState<CsvTable>();
  const [json, setJson] = useState<ImportedDataset>();
  const [roles, setRoles] = useState<ColumnRole[]>([]);
  const [oneHot, setOneHot] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const name = fileName.replace(/\.[^.]+$/, '') || t('sandbox.import.untitled');
  const info = useMemo(() => (table ? columnInfo(table) : []), [table]);
  const targets = roles.filter((r) => r === 't').length;
  const mapped = useMemo((): { data?: ImportedDataset; error?: string } => {
    if (json) return { data: json };
    if (!table) return {};
    try {
      return { data: mapCsv(table, mappingFromRoles(roles, oneHot), name) };
    } catch (error) {
      return { error: errorText(t, error) };
    }
  }, [json, table, roles, oneHot, name]);

  const load = async (file: File) => {
    setReadError(undefined);
    setTable(undefined);
    setJson(undefined);
    setFileName(file.name);
    try {
      const text = await readFile(file);
      const base = file.name.replace(/\.[^.]+$/, '');
      if (/\.json$/i.test(file.name) || text.trimStart().startsWith('{')) {
        setJson(parseJsonDataset(text, base));
      } else {
        const parsed = parseCsv(text);
        const guess = guessMapping(parsed);
        setTable(parsed);
        setRoles(guess.roles);
        setOneHot(guess.oneHot);
      }
    } catch (error) {
      setReadError(errorText(t, error));
    }
  };
  const setRole = (column: number, role: ColumnRole) =>
    setRoles((current) => {
      const next = current.map((r, c) => (c === column ? role : r));
      // One-hot needs exactly one target column; switch it off rather than fail.
      if (next.filter((r) => r === 't').length !== 1) setOneHot(false);
      return next;
    });
  const data = mapped.data;
  const rows = data ? { x: data.x, t: data.t } : undefined;
  const plot = rows && data ? rowsPlot(rows, data.task) : undefined;
  const loaded = !!(table || json);

  return (
    <Dialog
      t={t}
      title={t('sandbox.import.title')}
      onClose={onCancel}
      wide
      footer={
        <>
          <button className="btn" onClick={onCancel}>
            {t('sandbox.cancel')}
          </button>
          <button
            className="btn btn-primary"
            disabled={!data}
            onClick={() => data && onConfirm(data)}
          >
            {t('sandbox.import.confirm')}
          </button>
        </>
      }
    >
      <input
        ref={input}
        type="file"
        accept=".csv,.json,text/csv,application/json"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = '';
          if (file) void load(file);
        }}
      />
      {!loaded ? (
        <button
          className={`sb-drop${dragging ? ' is-over' : ''}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) void load(file);
          }}
        >
          <span className="sb-drop-icon" aria-hidden="true">
            ⇪
          </span>
          <span className="sb-drop-title">{t('sandbox.import.drop')}</span>
          <span className="sb-drop-hint">{t('sandbox.import.formats')}</span>
        </button>
      ) : (
        <div className="sb-file">
          <span className="sb-file-name">{fileName}</span>
          <span className="sb-file-meta">
            {table
              ? t('sandbox.import.tableSize', {
                  rows: table.rows.length,
                  columns: table.header.length,
                })
              : t('sandbox.import.jsonSize', { rows: json!.x.length })}
          </span>
          <button className="btn btn-small" onClick={() => input.current?.click()}>
            {t('sandbox.import.another')}
          </button>
        </div>
      )}
      {readError && (
        <p className="sb-message is-bad" role="alert">
          {readError}
        </p>
      )}
      {table && (
        <>
          <p className="sb-step">{t('sandbox.import.mapHelp')}</p>
          <div className="sb-map-scroll">
            <table className="sb-map">
              <thead>
                <tr>
                  {table.header.map((column, c) => (
                    <th key={c} className={`is-${roles[c]}`} scope="col">
                      <span className="sb-col-name" title={column}>
                        {column}
                      </span>
                      <span className="sb-col-kind">
                        {info[c]?.numeric ? t('sandbox.import.numeric') : t('sandbox.import.text')}
                      </span>
                      <span
                        className="sb-role"
                        role="radiogroup"
                        aria-label={t('sandbox.import.roleFor', { column })}
                      >
                        {ROLES.map((role) => (
                          <button
                            key={role}
                            role="radio"
                            aria-checked={roles[c] === role}
                            className={`sb-role-btn is-${role}${roles[c] === role ? ' is-on' : ''}`}
                            title={t(`sandbox.import.role.${role}.help`)}
                            onClick={() => setRole(c, role)}
                          >
                            {t(`sandbox.import.role.${role}`)}
                          </button>
                        ))}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.slice(0, PREVIEW_ROWS).map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c} className={`is-${roles[c]}`}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {table.rows.length > PREVIEW_ROWS && (
            <p className="hint">
              {t('sandbox.import.more', { n: table.rows.length - PREVIEW_ROWS })}
            </p>
          )}
          <label className={`sb-check${targets !== 1 ? ' is-disabled' : ''}`}>
            <input
              type="checkbox"
              checked={oneHot}
              disabled={targets !== 1}
              onChange={(e) => setOneHot(e.target.checked)}
            />
            <span>
              {t('sandbox.import.oneHot')}
              <span className="hint"> {t('sandbox.import.oneHotHelp')}</span>
            </span>
          </label>
        </>
      )}
      {loaded && (
        <div className={`sb-result${mapped.error ? ' is-bad' : ''}`} aria-live="polite">
          {data ? (
            <>
              {plot ? (
                <DataPlot data={plot} label={t('sandbox.import.preview')} />
              ) : (
                <p className="sb-empty-note">{t('sandbox.import.noPlot')}</p>
              )}
              <div>
                <h3 className="sb-block-title">{t('sandbox.import.result')}</h3>
                <dl className="kv sb-kv">
                  <dt>{t('sandbox.dataset.rows')}</dt>
                  <dd>{formatNumber(locale, data.x.length)}</dd>
                  <dt>{t('sandbox.dataset.input')}</dt>
                  <dd>x [{importedShapes(data).x.join('×')}]</dd>
                  <dt>{t('sandbox.dataset.target')}</dt>
                  <dd>t [{importedShapes(data).t.join('×')}]</dd>
                  <dt>{t('sandbox.import.task')}</dt>
                  <dd>
                    {t(`sandbox.task.${data.task}`)}
                    {data.task === 'classification'
                      ? ` · ${t('sandbox.dataset.classes', { n: data.t[0].length, count: data.t[0].length })}`
                      : ''}
                  </dd>
                </dl>
                {data.task === 'classification' && data.tColumns && (
                  <p className="hint">
                    {t('sandbox.import.classList', { list: data.tColumns.slice(0, 8).join(', ') })}
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="sb-message is-bad" role="alert">
              {mapped.error}
            </p>
          )}
        </div>
      )}
      <p className="hint">{t('sandbox.dataset.localOnly')}</p>
    </Dialog>
  );
}
