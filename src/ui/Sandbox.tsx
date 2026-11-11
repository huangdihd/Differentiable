// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Sandbox: local slots, a free canvas on a built-in or imported dataset, and weight files.
 * Nothing leaves the browser; files are read with FileReader and written as downloads.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { Progress, TrainLevel } from '../engine/levels';
import type { WorkerClient } from '../worker/client';
import { WorkerClientError } from '../worker/client';
import {
  DatasetImportError,
  importedShapes,
  mapCsv,
  parseCsv,
  parseJsonDataset,
  type CsvTable,
  type ImportedDataset,
} from '../sandbox/dataset-import';
import { buildSandboxLevel, datasetKey } from '../sandbox/level';
import {
  MAX_SLOTS,
  SANDBOX_BUILTINS,
  createSlot,
  exportWeights,
  parseWeights,
  removeSlot,
  sandboxLevelId,
  sandboxSlots,
  updateSlot,
  type SandboxDataset,
  type SandboxSlot,
} from '../sandbox/slots';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const newId = () => {
  try {
    return crypto.randomUUID();
  } catch {
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
};

const readFile = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });

function download(name: string, value: unknown) {
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
  dataset.kind === 'builtin'
    ? t(`datasets.${dataset.id}.name`)
    : t('sandbox.dataset.imported', { name: dataset.name });

/** Graph and training settings live in the slot; the Editor sees them as a level's own. */
function editorProgress(progress: Progress, slot: SandboxSlot, levelId: string): Progress {
  return {
    ...progress,
    savedGraphs: slot.graph
      ? { ...progress.savedGraphs, [levelId]: slot.graph }
      : progress.savedGraphs,
    ...(slot.training || progress.trainingSettings
      ? {
          trainingSettings: {
            ...progress.trainingSettings,
            ...(slot.training ? { [levelId]: slot.training } : {}),
          },
        }
      : {}),
  };
}

/** Move the Editor's per-level writes back into the slot and out of curriculum progress. */
export function fromEditorProgress(next: Progress, slotId: string, now: number): Progress {
  const levelId = sandboxLevelId(slotId);
  const graph = next.savedGraphs[levelId];
  const training = next.trainingSettings?.[levelId];
  const savedGraphs = { ...next.savedGraphs };
  delete savedGraphs[levelId];
  const stripped: Progress = { ...next, savedGraphs };
  if (next.trainingSettings) {
    const trainingSettings = { ...next.trainingSettings };
    delete trainingSettings[levelId];
    stripped.trainingSettings = trainingSettings;
  }
  return updateSlot(
    stripped,
    slotId,
    { ...(graph ? { graph } : {}), ...(training ? { training } : {}) },
    now,
  );
}

export function SandboxScreen({
  t,
  progress,
  onProgress,
  ready,
  client,
  onExit,
  renderEditor,
}: {
  t: T;
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
    sandbox: { label: string; panel: ReactNode };
  }) => ReactNode;
}) {
  const [open, setOpen] = useState<string>();
  const slots = sandboxSlots(progress);
  const slot = slots.find((s) => s.id === open);
  if (!slot)
    return (
      <SlotList
        t={t}
        slots={slots}
        ready={ready}
        onOpen={setOpen}
        onExit={onExit}
        onCreate={() => {
          const id = newId();
          onProgress((current) =>
            createSlot(current, {
              id,
              name: t('sandbox.slot.defaultName', { n: sandboxSlots(current).length + 1 }),
              now: Date.now(),
            }),
          );
          setOpen(id);
        }}
        onDelete={(id) => onProgress((current) => removeSlot(current, id))}
      />
    );
  return (
    <OpenSlot
      t={t}
      slot={slot}
      progress={progress}
      onProgress={onProgress}
      client={client}
      onBack={() => setOpen(undefined)}
      renderEditor={renderEditor}
    />
  );
}

function SlotList({
  t,
  slots,
  ready,
  onOpen,
  onCreate,
  onDelete,
  onExit,
}: {
  t: T;
  slots: SandboxSlot[];
  ready: boolean;
  onOpen: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onExit: () => void;
}) {
  const sorted = [...slots].sort((a, b) => b.updatedAt - a.updatedAt);
  return (
    <main className="home blueprint-bg">
      <div className="home-inner">
        <section className="chapter sandbox-slots">
          <header className="chapter-head">
            <span className="road-n">∞</span>
            <h2>{t('sandbox.title')}</h2>
            <p className="muted">{t('sandbox.slotsLead')}</p>
          </header>
          <div className="chapter-levels">
            <button
              className="level-card is-playable"
              disabled={!ready || slots.length >= MAX_SLOTS}
              onClick={onCreate}
            >
              <span className="level-id">+</span>
              <span className="level-card-title">{t('sandbox.slot.new')}</span>
              <span className="level-state">
                {slots.length >= MAX_SLOTS
                  ? t('sandbox.slot.full', { max: MAX_SLOTS })
                  : t('sandbox.slot.newHint')}
              </span>
            </button>
            {sorted.map((slot, i) => (
              <div key={slot.id} className="sandbox-slot">
                <button
                  className="level-card is-playable"
                  disabled={!ready}
                  onClick={() => onOpen(slot.id)}
                >
                  <span className="level-id">∞{i + 1}</span>
                  <span className="level-card-title">{slot.name}</span>
                  <span className="level-state">
                    {datasetLabel(t, slot.dataset)} ·{' '}
                    {t('sandbox.slot.updated', {
                      date: new Date(slot.updatedAt).toLocaleString(),
                    })}
                  </span>
                </button>
                <button
                  type="button"
                  className="icon-btn sandbox-slot-delete"
                  aria-label={t('sandbox.slot.delete')}
                  title={t('sandbox.slot.delete')}
                  onClick={() => {
                    if (window.confirm(t('sandbox.slot.deleteConfirm', { name: slot.name })))
                      onDelete(slot.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <p>
            <button className="btn" onClick={onExit}>
              {t('sandbox.back')}
            </button>
          </p>
        </section>
      </div>
    </main>
  );
}

function OpenSlot({
  t,
  slot,
  progress,
  onProgress,
  client,
  onBack,
  renderEditor,
}: {
  t: T;
  slot: SandboxSlot;
  progress: Progress;
  onProgress: (update: (current: Progress) => Progress) => void;
  client: React.RefObject<WorkerClient | null>;
  onBack: () => void;
  renderEditor: Parameters<typeof SandboxScreen>[0]['renderEditor'];
}) {
  const levelId = sandboxLevelId(slot.id);
  const dataKey = datasetKey(slot.dataset);
  const weightsKey = slot.weights ? JSON.stringify(slot.weights).length : 0;
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
  const [message, setMessage] = useState<string>();
  const [importing, setImporting] = useState(false);
  const current = useRef(slot);
  current.current = slot;
  useEffect(() => {
    let alive = true;
    setSynced(undefined);
    const s = current.current;
    client.current
      ?.setSandbox({
        slotId: s.id,
        dataset: s.dataset,
        ...(s.weights ? { weights: s.weights } : {}),
      })
      .then(
        () => alive && setSynced(syncKey),
        (error) => alive && setMessage(errorText(t, error)),
      );
    return () => {
      alive = false;
    };
  }, [syncKey]);
  const change = (update: Partial<Omit<SandboxSlot, 'id' | 'createdAt'>>) =>
    onProgress((p) => updateSlot(p, slot.id, update, Date.now()));
  const weightsInput = useRef<HTMLInputElement>(null);

  const panel = (
    <div className="sandbox-panel">
      <div className="sandbox-panel-block">
        <label className="label" htmlFor="sandbox-name">
          {t('sandbox.slot.name')}
        </label>
        <input
          id="sandbox-name"
          className="field"
          value={slot.name}
          maxLength={60}
          onChange={(e) => change({ name: e.target.value })}
        />
      </div>
      <div className="sandbox-panel-block">
        <label className="label" htmlFor="sandbox-dataset">
          {t('sandbox.dataset.label')}
        </label>
        <select
          id="sandbox-dataset"
          className="field"
          value={slot.dataset.kind === 'builtin' ? slot.dataset.id : '__imported'}
          onChange={(e) => {
            if (e.target.value === '__import') setImporting(true);
            else if (e.target.value !== '__imported')
              change({
                dataset: {
                  kind: 'builtin',
                  id: e.target.value as (typeof SANDBOX_BUILTINS)[number],
                },
              });
          }}
        >
          {SANDBOX_BUILTINS.map((id) => (
            <option key={id} value={id}>
              {t(`datasets.${id}.name`)}
            </option>
          ))}
          {slot.dataset.kind === 'imported' && (
            <option value="__imported">{datasetLabel(t, slot.dataset)}</option>
          )}
          <option value="__import">{t('sandbox.dataset.importOption')}</option>
        </select>
        <p className="hint">{t('sandbox.dataset.localOnly')}</p>
      </div>
      <div className="sandbox-panel-block">
        <span className="label">{t('sandbox.weights.label')}</span>
        <div className="btn-row">
          <button
            className="btn"
            onClick={() => {
              setMessage(undefined);
              client.current
                ?.exportWeights(levelId, slot.graph ?? { version: 1, nodes: [], edges: [] })
                .then(({ parameters }) => {
                  download(`${slot.name || 'sandbox'}.weights.json`, exportWeights(parameters));
                  setMessage(t('sandbox.weights.exported'));
                })
                .catch((error) => setMessage(errorText(t, error)));
            }}
          >
            {t('sandbox.weights.export')}
          </button>
          <button className="btn" onClick={() => weightsInput.current?.click()}>
            {t('sandbox.weights.import')}
          </button>
          {slot.weights && (
            <button className="btn" onClick={() => change({ weights: undefined })}>
              {t('sandbox.weights.clear')}
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
              change({ weights });
              setMessage(
                t('sandbox.weights.imported', { n: Object.keys(weights.parameters).length }),
              );
            } catch (error) {
              setMessage(errorText(t, error));
            }
          }}
        />
        {slot.weights && (
          <p className="hint">
            {t('sandbox.weights.active', { n: Object.keys(slot.weights.parameters).length })}
          </p>
        )}
      </div>
      {message && (
        <p className="hint" role="status">
          {message}
        </p>
      )}
      <div className="btn-row">
        <button className="btn" onClick={onBack}>
          {t('sandbox.backToSlots')}
        </button>
      </div>
    </div>
  );

  if (importing)
    return (
      <main className="home blueprint-bg">
        <div className="home-inner">
          <ImportWizard
            t={t}
            onCancel={() => setImporting(false)}
            onConfirm={(data) => {
              change({ dataset: { kind: 'imported', ...data } });
              setImporting(false);
            }}
          />
        </div>
      </main>
    );
  if (level instanceof Error)
    return (
      <main className="home blueprint-bg">
        <div className="home-inner">
          <p className="status status-fail">{errorText(t, level)}</p>
          <button className="btn" onClick={onBack}>
            {t('sandbox.backToSlots')}
          </button>
        </div>
      </main>
    );
  if (synced !== syncKey)
    return (
      <main className="home blueprint-bg">
        <div className="home-inner" role="status">
          {message ?? t('sandbox.loading')}
        </div>
      </main>
    );
  return renderEditor({
    key: syncKey,
    level,
    progress: editorProgress(progress, slot, levelId),
    onProgress: (next) => onProgress(() => fromEditorProgress(next, slot.id, Date.now())),
    sandbox: { label: t('sandbox.label'), panel },
  });
}

const PREVIEW_ROWS = 5;

/** Choose file → map columns (CSV only) → preview and confirm. */
export function ImportWizard({
  t,
  onCancel,
  onConfirm,
}: {
  t: T;
  onCancel: () => void;
  onConfirm: (data: ImportedDataset) => void;
}) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [error, setError] = useState<string>();
  const [fileName, setFileName] = useState('');
  const [table, setTable] = useState<CsvTable>();
  const [x, setX] = useState<number[]>([]);
  const [target, setTarget] = useState<number[]>([]);
  const [oneHot, setOneHot] = useState(false);
  const [data, setData] = useState<ImportedDataset>();
  const name = fileName.replace(/\.[^.]+$/, '') || t('sandbox.import.untitled');
  const toggle = (list: number[], set: (v: number[]) => void, column: number) =>
    set(list.includes(column) ? list.filter((c) => c !== column) : [...list, column].sort());
  return (
    <section className="chapter sandbox-import">
      <header className="chapter-head">
        <span className="road-n">{t('sandbox.import.step', { n: step, total: 3 })}</span>
        <h2>{t(`sandbox.import.title${step}`)}</h2>
        <p className="muted">{t('sandbox.dataset.localOnly')}</p>
      </header>
      {step === 1 && (
        <>
          <p>{t('sandbox.import.formats')}</p>
          <input
            type="file"
            accept=".csv,.json,text/csv,application/json"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              setError(undefined);
              setFileName(file.name);
              try {
                const text = await readFile(file);
                const base = file.name.replace(/\.[^.]+$/, '');
                if (/\.json$/i.test(file.name) || text.trimStart().startsWith('{')) {
                  setData(parseJsonDataset(text, base));
                  setStep(3);
                } else {
                  const parsed = parseCsv(text);
                  setTable(parsed);
                  const last = parsed.header.length - 1;
                  setX(parsed.header.map((_, i) => i).filter((i) => i !== last));
                  setTarget([last]);
                  setOneHot(false);
                  setStep(2);
                }
              } catch (cause) {
                setError(errorText(t, cause));
              }
            }}
          />
        </>
      )}
      {step === 2 && table && (
        <>
          <p>{t('sandbox.import.mapHelp')}</p>
          <table className="sandbox-map">
            <thead>
              <tr>
                <th>{t('sandbox.import.column')}</th>
                <th>x</th>
                <th>t</th>
                <th>{t('sandbox.import.sample')}</th>
              </tr>
            </thead>
            <tbody>
              {table.header.map((column, i) => (
                <tr key={i}>
                  <td>{column}</td>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`x ${column}`}
                      checked={x.includes(i)}
                      onChange={() => {
                        toggle(x, setX, i);
                        setTarget((v) => v.filter((c) => c !== i));
                      }}
                    />
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`t ${column}`}
                      checked={target.includes(i)}
                      onChange={() => {
                        toggle(target, setTarget, i);
                        setX((v) => v.filter((c) => c !== i));
                      }}
                    />
                  </td>
                  <td className="muted">{table.rows[0]?.[i]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <label>
            <input
              type="checkbox"
              checked={oneHot}
              disabled={target.length !== 1}
              onChange={(e) => setOneHot(e.target.checked)}
            />{' '}
            {t('sandbox.import.oneHot')}
          </label>
          <p>
            <button className="btn" onClick={() => setStep(1)}>
              {t('sandbox.import.back')}
            </button>{' '}
            <button
              className="btn btn-primary"
              onClick={() => {
                setError(undefined);
                try {
                  setData(
                    mapCsv(table, { x, t: target, oneHot: oneHot && target.length === 1 }, name),
                  );
                  setStep(3);
                } catch (cause) {
                  setError(errorText(t, cause));
                }
              }}
            >
              {t('sandbox.import.next')}
            </button>
          </p>
        </>
      )}
      {step === 3 && data && (
        <>
          <dl className="kv">
            <dt>{t('sandbox.import.rows')}</dt>
            <dd>{data.x.length}</dd>
            <dt>{t('sandbox.import.task')}</dt>
            <dd>{t(`sandbox.import.task.${data.task}`)}</dd>
            <dt>{t('sandbox.import.modality')}</dt>
            <dd>{t(`datasets.modality.${data.modality}`)}</dd>
            <dt>{t('sandbox.import.shapes')}</dt>
            <dd>
              x [{importedShapes(data).x.join('×')}] · t [{importedShapes(data).t.join('×')}]
            </dd>
          </dl>
          <table className="sandbox-map">
            <thead>
              <tr>
                {(data.xColumns ?? data.x[0].map((_, i) => `x${i + 1}`)).map((c, i) => (
                  <th key={`x${i}`}>x · {c}</th>
                ))}
                {(data.tColumns ?? data.t[0].map((_, i) => `t${i + 1}`)).map((c, i) => (
                  <th key={`t${i}`}>t · {c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.x.slice(0, PREVIEW_ROWS).map((row, i) => (
                <tr key={i}>
                  {[...row, ...data.t[i]].map((v, j) => (
                    <td key={j}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {data.x.length > PREVIEW_ROWS && (
            <p className="muted">{t('sandbox.import.more', { n: data.x.length - PREVIEW_ROWS })}</p>
          )}
          <p>
            <button className="btn" onClick={() => setStep(table ? 2 : 1)}>
              {t('sandbox.import.back')}
            </button>{' '}
            <button className="btn btn-primary" onClick={() => onConfirm(data)}>
              {t('sandbox.import.confirm')}
            </button>
          </p>
        </>
      )}
      {error && (
        <p className="status status-fail" role="alert">
          {error}
        </p>
      )}
      <p>
        <button className="btn" onClick={onCancel}>
          {t('sandbox.import.cancel')}
        </button>
      </p>
    </section>
  );
}
