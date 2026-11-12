// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * "My components" at the top of the sandbox library: parts the player packaged, placed like any
 * other component, with rename / export / delete on each and file import for the whole shelf.
 */
import { useRef, useState, type PointerEvent } from 'react';
import {
  MAX_COMPONENT_NAME,
  customComponentName,
  type CustomComponent,
} from '../sandbox/components';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const shapeText = (t: T, shape?: readonly number[]) =>
  shape ? (shape.length ? `[${shape.join('×')}]` : t('ui.scalar')) : '';

/** "x [40] · t [40] → out [40]" for the tooltip. */
export function portSummary(t: T, component: CustomComponent): string {
  const side = (ports: readonly string[]) =>
    ports
      .map((p) => `${p}${component.shapes?.[p] ? ` ${shapeText(t, component.shapes[p])}` : ''}`)
      .join(' · ') || '—';
  return `${side(component.inputs)} → ${side(component.outputs)}`;
}

export function MyComponents({
  t,
  library,
  disabled,
  usage,
  onPress,
  onActivate,
  onRename,
  onDelete,
  onExport,
  onImport,
}: {
  t: T;
  library: readonly CustomComponent[];
  disabled: boolean;
  usage: (id: string) => { slots: string[]; components: string[] };
  onPress: (component: string, event: PointerEvent) => void;
  onActivate: (component: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  onExport: (ids: string[]) => void;
  onImport: (file: File) => Promise<string>;
}) {
  const [open, setOpen] = useState(true);
  const [menu, setMenu] = useState<string>();
  const [renaming, setRenaming] = useState<{ id: string; draft: string }>();
  const [confirming, setConfirming] = useState<string>();
  const [message, setMessage] = useState<{ text: string; bad?: boolean }>();
  const file = useRef<HTMLInputElement>(null);
  const sorted = [...library].sort((a, b) => a.createdAt - b.createdAt);
  return (
    <details
      className="palette-group my-components"
      data-category="custom"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span>{t('sandbox.components.title')}</span>
        <span className="palette-count">{library.length}</span>
      </summary>
      <div className="my-components-body">
        {sorted.length === 0 ? (
          <p className="my-components-empty">{t('sandbox.components.empty')}</p>
        ) : (
          <ul className="my-components-list">
            {sorted.map((c) => {
              const name = customComponentName(c.id);
              const used = usage(c.id);
              const blocked = used.slots.length + used.components.length > 0;
              return (
                <li key={c.id} className={`my-component${menu === c.id ? ' is-open' : ''}`}>
                  {renaming?.id === c.id ? (
                    <form
                      className="my-component-rename"
                      onSubmit={(e) => {
                        e.preventDefault();
                        onRename(c.id, renaming.draft);
                        setRenaming(undefined);
                        setMenu(undefined);
                      }}
                    >
                      <input
                        className="field"
                        autoFocus
                        value={renaming.draft}
                        maxLength={MAX_COMPONENT_NAME}
                        aria-label={t('sandbox.components.name')}
                        onChange={(e) => setRenaming({ id: c.id, draft: e.target.value })}
                        onKeyDown={(e) => e.key === 'Escape' && setRenaming(undefined)}
                      />
                      <button
                        className="btn btn-primary btn-small"
                        disabled={!renaming.draft.trim()}
                      >
                        {t('sandbox.slot.save')}
                      </button>
                    </form>
                  ) : (
                    <div className="my-component-row">
                      <button
                        type="button"
                        className="palette-item is-player"
                        data-component={name}
                        data-category="custom"
                        disabled={disabled}
                        title={`${c.name}\n${portSummary(t, c)}\n${t('sandbox.components.dragHint')}`}
                        onPointerDown={(event) => onPress(name, event)}
                        onClick={(event) => event.detail === 0 && onActivate(name)}
                      >
                        <span className="sym">⊞</span>
                        <span className="name">{c.name}</span>
                        <span className="my-component-ports">
                          {t('sandbox.components.ports', {
                            inputs: c.inputs.length,
                            outputs: c.outputs.length,
                          })}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="icon-btn my-component-more"
                        aria-expanded={menu === c.id}
                        aria-label={t('sandbox.components.more', { name: c.name })}
                        title={t('sandbox.components.more', { name: c.name })}
                        onClick={() => {
                          setConfirming(undefined);
                          setMenu(menu === c.id ? undefined : c.id);
                        }}
                      >
                        ⋯
                      </button>
                    </div>
                  )}
                  {menu === c.id && !renaming && (
                    <div className="my-component-menu">
                      {confirming === c.id ? (
                        <div className="sb-confirm" role="alertdialog">
                          {blocked ? (
                            <span>
                              {t('sandbox.components.inUse', {
                                name: c.name,
                                where: [...used.slots, ...used.components]
                                  .map((n) => `「${n}」`)
                                  .join(t('sandbox.components.listSep')),
                              })}
                            </span>
                          ) : (
                            <span>{t('sandbox.components.deleteConfirm', { name: c.name })}</span>
                          )}
                          <span className="btn-row">
                            {!blocked && (
                              <button
                                className="btn btn-small btn-danger is-armed"
                                autoFocus
                                onClick={() => {
                                  onDelete(c.id);
                                  setMenu(undefined);
                                  setConfirming(undefined);
                                }}
                              >
                                {t('sandbox.slot.delete')}
                              </button>
                            )}
                            <button
                              className="btn btn-small"
                              onClick={() => setConfirming(undefined)}
                            >
                              {blocked ? t('sandbox.close') : t('sandbox.cancel')}
                            </button>
                          </span>
                        </div>
                      ) : (
                        <div className="btn-row">
                          <button
                            className="btn btn-small"
                            onClick={() => setRenaming({ id: c.id, draft: c.name })}
                          >
                            {t('sandbox.slot.rename')}
                          </button>
                          <button
                            className="btn btn-small"
                            onClick={() => {
                              onExport([c.id]);
                              setMessage({ text: t('sandbox.components.exported') });
                              setMenu(undefined);
                            }}
                          >
                            ⇩ {t('sandbox.components.export')}
                          </button>
                          <button
                            className="btn btn-small btn-danger"
                            onClick={() => setConfirming(c.id)}
                          >
                            {t('sandbox.slot.delete')}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        <div className="my-components-files">
          <button
            className="btn btn-small"
            disabled={disabled}
            onClick={() => file.current?.click()}
          >
            ⇧ {t('sandbox.components.import')}
          </button>
          {library.length > 0 && (
            <button
              className="btn btn-small"
              onClick={() => {
                onExport(library.map((c) => c.id));
                setMessage({ text: t('sandbox.components.exported') });
              }}
            >
              ⇩ {t('sandbox.components.exportAll')}
            </button>
          )}
        </div>
        <input
          ref={file}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={async (e) => {
            const chosen = e.target.files?.[0];
            e.target.value = '';
            if (!chosen) return;
            try {
              setMessage({ text: await onImport(chosen) });
            } catch (error) {
              setMessage({
                text: t(
                  error instanceof Error && error.message.startsWith('sandbox.')
                    ? error.message
                    : 'sandbox.import.error.read',
                ),
                bad: true,
              });
            }
          }}
        />
        {message && (
          <p className={`sb-message${message.bad ? ' is-bad' : ''}`} role="status">
            {message.text}
          </p>
        )}
      </div>
    </details>
  );
}
