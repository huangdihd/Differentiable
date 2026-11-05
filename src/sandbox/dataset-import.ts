// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Local dataset import for the sandbox. Everything here runs in the browser on text the player
 * picked; nothing is uploaded. Errors are i18n keys (`sandbox.import.error.*`) plus params.
 */

export type ImportedModality = 'line' | 'points' | 'table';
export type ImportedTask = 'regression' | 'classification';

/** The imported rows, ready to become a TrainingBatch (`inputs.x` / `inputs.t`). */
export interface ImportedDataset {
  name: string;
  modality: ImportedModality;
  task: ImportedTask;
  /** One row per sample. */
  x: number[][];
  t: number[][];
  xColumns?: string[];
  tColumns?: string[];
}

export class DatasetImportError extends Error {
  constructor(
    readonly key: string,
    readonly params: Record<string, string | number> = {},
  ) {
    super(key);
  }
}

/** Storage stays small: localStorage is a few megabytes per site. */
export const MAX_ROWS = 2000;
export const MAX_COLUMNS = 32;
export const MAX_CLASSES = 32;

export interface CsvTable {
  header: string[];
  rows: string[][];
}

/** RFC 4180-ish: quoted fields, doubled quotes, comma or semicolon separators, CRLF. */
export function parseCsv(text: string): CsvTable {
  const source = text.replace(/^\uFEFF/, '');
  const firstLine = source.split(/\r?\n/, 1)[0] ?? '';
  const separator = !firstLine.includes(',') && firstLine.includes(';') ? ';' : ',';
  const records: string[][] = [];
  let field = '';
  let record: string[] = [];
  let quoted = false;
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (quoted) {
      if (c === '"' && source[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"' && field === '') quoted = true;
    else if (c === separator) {
      record.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && source[i + 1] === '\n') i++;
      record.push(field);
      records.push(record);
      record = [];
      field = '';
    } else field += c;
  }
  if (quoted) throw new DatasetImportError('sandbox.import.error.unterminatedQuote');
  if (field !== '' || record.length) {
    record.push(field);
    records.push(record);
  }
  const nonEmpty = records.filter((r) => r.some((cell) => cell.trim() !== ''));
  if (nonEmpty.length < 2) throw new DatasetImportError('sandbox.import.error.empty');
  const header = nonEmpty[0].map((h, i) => h.trim() || `#${i + 1}`);
  if (header.length > MAX_COLUMNS + 1)
    throw new DatasetImportError('sandbox.import.error.tooManyColumns', { max: MAX_COLUMNS });
  const rows = nonEmpty.slice(1);
  if (rows.length > MAX_ROWS)
    throw new DatasetImportError('sandbox.import.error.tooManyRows', { max: MAX_ROWS });
  rows.forEach((row, i) => {
    if (row.length !== header.length)
      throw new DatasetImportError('sandbox.import.error.ragged', {
        row: i + 2,
        expected: header.length,
        actual: row.length,
      });
  });
  return { header, rows: rows.map((row) => row.map((cell) => cell.trim())) };
}

const number = (cell: string, row: number, column: string) => {
  const value = cell === '' ? NaN : Number(cell);
  if (!Number.isFinite(value))
    throw new DatasetImportError('sandbox.import.error.notNumber', { row, column, value: cell });
  return value;
};

export interface CsvMapping {
  x: readonly number[];
  t: readonly number[];
  /** Classification: one column of class ids (any labels) becomes one-hot rows. */
  oneHot?: boolean;
}

/** Class labels in first-seen order; numeric labels sort numerically so 0,1,2 stay in place. */
export function classLabels(values: readonly string[]): string[] {
  const labels = [...new Set(values)];
  return labels.every((v) => v !== '' && Number.isFinite(Number(v)))
    ? labels.sort((a, b) => Number(a) - Number(b))
    : labels;
}

export function oneHot(values: readonly string[]): { labels: string[]; rows: number[][] } {
  const labels = classLabels(values);
  if (labels.length < 2) throw new DatasetImportError('sandbox.import.error.oneClass');
  if (labels.length > MAX_CLASSES)
    throw new DatasetImportError('sandbox.import.error.tooManyClasses', { max: MAX_CLASSES });
  const index = new Map(labels.map((label, i) => [label, i]));
  return {
    labels,
    rows: values.map((v) => labels.map((_, i) => (index.get(v) === i ? 1 : 0))),
  };
}

const modalityFor = (task: ImportedTask, xWidth: number, tWidth: number): ImportedModality =>
  task === 'regression' && xWidth === 1 && tWidth === 1
    ? 'line'
    : task === 'classification' && xWidth === 2
      ? 'points'
      : 'table';

/** Classification when the target is one-hot (every row a single 1, rest 0). */
function looksOneHot(t: readonly number[][]): boolean {
  return (
    t[0].length >= 2 &&
    t.every(
      (row) => row.every((v) => v === 0 || v === 1) && row.filter((v) => v === 1).length === 1,
    )
  );
}

export function mapCsv(table: CsvTable, mapping: CsvMapping, name: string): ImportedDataset {
  const columns = table.header.length;
  const valid = (list: readonly number[]) =>
    list.every((c) => Number.isInteger(c) && c >= 0 && c < columns) &&
    new Set(list).size === list.length;
  if (!mapping.x.length) throw new DatasetImportError('sandbox.import.error.noX');
  if (!mapping.t.length) throw new DatasetImportError('sandbox.import.error.noT');
  if (!valid(mapping.x) || !valid(mapping.t) || mapping.x.some((c) => mapping.t.includes(c)))
    throw new DatasetImportError('sandbox.import.error.overlap');
  if (mapping.oneHot && mapping.t.length !== 1)
    throw new DatasetImportError('sandbox.import.error.oneHotColumn');
  const x = table.rows.map((row, i) =>
    mapping.x.map((c) => number(row[c], i + 2, table.header[c])),
  );
  let t: number[][];
  let tColumns = mapping.t.map((c) => table.header[c]);
  if (mapping.oneHot) {
    const encoded = oneHot(table.rows.map((row) => row[mapping.t[0]]));
    t = encoded.rows;
    tColumns = encoded.labels;
  } else
    t = table.rows.map((row, i) => mapping.t.map((c) => number(row[c], i + 2, table.header[c])));
  const task: ImportedTask = mapping.oneHot || looksOneHot(t) ? 'classification' : 'regression';
  return {
    name,
    task,
    modality: modalityFor(task, x[0].length, t[0].length),
    x,
    t,
    xColumns: mapping.x.map((c) => table.header[c]),
    tColumns,
  };
}

const matrix = (value: unknown): value is number[][] =>
  Array.isArray(value) &&
  value.every((row) => Array.isArray(row) && row.every((v) => typeof v === 'number'));

/** Shape and content checks shared by JSON import and saved slots. */
export function validateRows(x: unknown, t: unknown): { x: number[][]; t: number[][] } {
  if (!matrix(x)) throw new DatasetImportError('sandbox.import.error.jsonMatrix', { field: 'x' });
  if (!matrix(t)) throw new DatasetImportError('sandbox.import.error.jsonMatrix', { field: 't' });
  if (x.length < 1) throw new DatasetImportError('sandbox.import.error.empty');
  if (x.length > MAX_ROWS)
    throw new DatasetImportError('sandbox.import.error.tooManyRows', { max: MAX_ROWS });
  if (x.length !== t.length)
    throw new DatasetImportError('sandbox.import.error.rowCount', { x: x.length, t: t.length });
  for (const [field, rows] of [
    ['x', x],
    ['t', t],
  ] as const) {
    const width = rows[0].length;
    if (width < 1 || width > MAX_COLUMNS)
      throw new DatasetImportError('sandbox.import.error.width', { field, max: MAX_COLUMNS });
    rows.forEach((row, i) => {
      if (row.length !== width)
        throw new DatasetImportError('sandbox.import.error.jsonRagged', {
          field,
          row: i + 1,
          expected: width,
          actual: row.length,
        });
      if (!row.every(Number.isFinite))
        throw new DatasetImportError('sandbox.import.error.jsonFinite', { field, row: i + 1 });
    });
  }
  return { x, t };
}

/** `{"modality": "points"|"table"|"line", "x": number[][], "t": number[][]}`. */
export function parseJsonDataset(text: string, name: string): ImportedDataset {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new DatasetImportError('sandbox.import.error.json');
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw new DatasetImportError('sandbox.import.error.jsonObject');
  const data = value as Record<string, unknown>;
  const { x, t } = validateRows(data.x, data.t);
  const task: ImportedTask = looksOneHot(t) ? 'classification' : 'regression';
  const modality = data.modality ?? modalityFor(task, x[0].length, t[0].length);
  if (modality !== 'line' && modality !== 'points' && modality !== 'table')
    throw new DatasetImportError('sandbox.import.error.modality', { value: String(modality) });
  if (modality === 'line' && (x[0].length !== 1 || t[0].length !== 1))
    throw new DatasetImportError('sandbox.import.error.lineShape');
  if (modality === 'points' && x[0].length !== 2)
    throw new DatasetImportError('sandbox.import.error.pointsShape');
  return { name, modality, task, x, t };
}

/**
 * Tensor shapes: a one-column curve stays a vector ([N] like the built-in noisy line, so the
 * regression plot works); everything else is a [N, columns] matrix.
 */
export function importedShapes(data: Pick<ImportedDataset, 'x' | 't'>): {
  x: number[];
  t: number[];
} {
  const n = data.x.length;
  const vector = data.x[0].length === 1 && data.t[0].length === 1;
  return vector ? { x: [n], t: [n] } : { x: [n, data.x[0].length], t: [n, data.t[0].length] };
}
