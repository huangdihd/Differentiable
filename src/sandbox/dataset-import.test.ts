// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  DatasetImportError,
  MAX_ROWS,
  guessMapping,
  importedShapes,
  mappingFromRoles,
  mapCsv,
  oneHot,
  parseCsv,
  parseJsonDataset,
} from './dataset-import';

const key = (run: () => unknown) => {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(DatasetImportError);
    return (error as DatasetImportError).key;
  }
  throw new Error('expected an import error');
};

describe('parseCsv', () => {
  it('reads a header, quotes, BOM and semicolons', () => {
    expect(parseCsv('\uFEFFa,"b, c"\n1,"2"\r\n3,4\n')).toEqual({
      header: ['a', 'b, c'],
      rows: [
        ['1', '2'],
        ['3', '4'],
      ],
    });
    expect(parseCsv('a;b\n1;2').header).toEqual(['a', 'b']);
  });
  it('rejects empty, ragged and unterminated input', () => {
    expect(key(() => parseCsv('a,b'))).toBe('sandbox.import.error.empty');
    expect(key(() => parseCsv('a,b\n1'))).toBe('sandbox.import.error.ragged');
    expect(key(() => parseCsv('a,b\n"1,2'))).toBe('sandbox.import.error.unterminatedQuote');
    const big = 'a\n' + Array.from({ length: MAX_ROWS + 1 }, () => '1').join('\n');
    expect(key(() => parseCsv(big))).toBe('sandbox.import.error.tooManyRows');
  });
});

describe('mapCsv', () => {
  const table = parseCsv('x1,x2,label\n0,1,cat\n1,0,dog\n2,2,cat');
  it('one-hot encodes a class column into points', () => {
    const data = mapCsv(table, { x: [0, 1], t: [2], oneHot: true }, 'pets');
    expect(data).toMatchObject({ modality: 'points', task: 'classification', name: 'pets' });
    expect(data.t).toEqual([
      [1, 0],
      [0, 1],
      [1, 0],
    ]);
    expect(importedShapes(data)).toEqual({ x: [3, 2], t: [3, 2] });
  });
  it('maps one x and one t to a line with [N] shapes', () => {
    const data = mapCsv(parseCsv('x,y\n1,2\n2,4'), { x: [0], t: [1] }, 'l');
    expect(data).toMatchObject({ modality: 'line', task: 'regression', x: [[1], [2]] });
    expect(importedShapes(data)).toEqual({ x: [2], t: [2] });
  });
  it('reports bad mappings and non-numbers', () => {
    expect(key(() => mapCsv(table, { x: [], t: [2] }, 'n'))).toBe('sandbox.import.error.noX');
    expect(key(() => mapCsv(table, { x: [0], t: [] }, 'n'))).toBe('sandbox.import.error.noT');
    expect(key(() => mapCsv(table, { x: [0], t: [0] }, 'n'))).toBe('sandbox.import.error.overlap');
    expect(key(() => mapCsv(table, { x: [0], t: [1, 2], oneHot: true }, 'n'))).toBe(
      'sandbox.import.error.oneHotColumn',
    );
    expect(key(() => mapCsv(table, { x: [0], t: [2] }, 'n'))).toBe(
      'sandbox.import.error.notNumber',
    );
  });
  it('needs at least two classes', () => {
    expect(oneHot(['a', 'b', 'a']).labels).toEqual(['a', 'b']);
    expect(key(() => oneHot(['a', 'a']))).toBe('sandbox.import.error.oneClass');
  });
});

describe('parseJsonDataset', () => {
  it('accepts a valid dataset', () => {
    const data = parseJsonDataset(
      JSON.stringify({ modality: 'table', x: [[1, 2, 3]], t: [[0, 1]] }),
      'j',
    );
    expect(data).toMatchObject({ modality: 'table', task: 'classification', name: 'j' });
  });
  it.each([
    ['nope', 'sandbox.import.error.json'],
    ['[]', 'sandbox.import.error.jsonObject'],
    ['{"modality":"text","x":[[1]],"t":[[1]]}', 'sandbox.import.error.modality'],
    ['{"modality":"line","x":[1],"t":[[1]]}', 'sandbox.import.error.jsonMatrix'],
    ['{"modality":"line","x":[[1],[2]],"t":[[1]]}', 'sandbox.import.error.rowCount'],
    ['{"modality":"table","x":[[1],[2,3]],"t":[[1],[2]]}', 'sandbox.import.error.jsonRagged'],
    ['{"modality":"line","x":[[1,2]],"t":[[1]]}', 'sandbox.import.error.lineShape'],
    ['{"modality":"points","x":[[1]],"t":[[1]]}', 'sandbox.import.error.pointsShape'],
  ])('rejects %s', (text, expected) => {
    expect(key(() => parseJsonDataset(text, 'j'))).toBe(expected);
  });
});

describe('guessMapping', () => {
  it('treats a named text label column as one-hot classes and skips other text', () => {
    const table = parseCsv('id,a,b,label\nr1,1,2,cat\nr2,3,4,dog\nr3,5,6,cat\n');
    expect(guessMapping(table)).toEqual({ roles: ['skip', 'x', 'x', 't'], oneHot: true });
  });
  it('reads a few whole-number targets as classes and a continuous one as regression', () => {
    const classes = parseCsv('x1,x2,y\n0,1,0\n1,0,1\n1,1,0\n0,0,1\n2,2,1\n');
    expect(guessMapping(classes).oneHot).toBe(true);
    const curve = parseCsv('x,t\n0,0.1\n1,0.9\n2,2.2\n3,2.8\n');
    expect(guessMapping(curve)).toEqual({ roles: ['x', 't'], oneHot: false });
  });
  it('falls back to the last column and maps roles to indices', () => {
    const table = parseCsv('a,b,c\n1,2,3.5\n4,5,6.5\n');
    const { roles, oneHot: classes } = guessMapping(table);
    expect(roles).toEqual(['x', 'x', 't']);
    expect(mappingFromRoles(roles, classes)).toEqual({ x: [0, 1], t: [2], oneHot: false });
    expect(mappingFromRoles(['t', 'x', 't'], true)).toEqual({ x: [1], t: [0, 2], oneHot: false });
  });
});
