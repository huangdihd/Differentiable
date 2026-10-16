/** Parse a finite decimal/scientific literal, optionally divided by one other literal. */
export function parseRealNumber(text: string): number {
  const literal = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
  const parts = text.trim().split('/');
  if (parts.length > 2) throw Error('Invalid real number');
  const numbers = parts.map((part) => {
    const token = part.trim();
    if (!literal.test(token)) throw Error('Invalid real number');
    const number = Number(token);
    if (!Number.isFinite(number)) throw Error('Invalid real number');
    return number;
  });
  if (numbers.length === 2 && numbers[1] === 0) throw Error('Zero denominator');
  const value = numbers.length === 2 ? numbers[0]! / numbers[1]! : numbers[0]!;
  if (!Number.isFinite(value)) throw Error('Invalid real number');
  return value;
}

/** Constants also accept flat bracketed lists, with the same syntax for each element. */
export function parseRealValue(text: string): number | number[] {
  const trimmed = text.trim();
  if (!trimmed.startsWith('[')) return parseRealNumber(trimmed);
  if (!trimmed.endsWith(']')) throw Error('Invalid real list');
  const contents = trimmed.slice(1, -1).trim();
  return contents ? contents.split(',').map(parseRealNumber) : [];
}
