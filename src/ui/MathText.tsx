import { Fragment, type ReactNode } from 'react';

/** Words set upright inside a formula: function names, not products of variables. */
const FUNCTIONS = new Set([
  'exp',
  'log',
  'ln',
  'tanh',
  'sigmoid',
  'softmax',
  'max',
  'min',
  'sum',
  'mean',
  'var',
  'relu',
  'ReLU',
  'sin',
  'cos',
  'sqrt',
  'onehot',
  'one_hot',
  'gather',
  'matmul',
  'LayerNorm',
  'Attention',
  'MSE',
  'CE',
]);

const SCRIPT = '[⁰-ⁿ₀-ₜᵢⱼᵀ′″ˣʸᵏⁱᵘ⁻⁺]*';
const TOKEN = new RegExp(
  [
    `(?<word>[A-Za-z_][A-Za-z_]+)${SCRIPT}`, // function names, or several letters
    `(?<ident>[A-Za-zα-ωΑ-Ωθσεμ∂∇])${SCRIPT}`, // one variable with its sub/superscripts
    `(?<num>\\d+(?:\\.\\d+)?${SCRIPT})`,
    `(?<op>[=≈≠≤≥<>+−\\-×·⊙/→←↦⇔∈])`,
    `(?<other>[\\s\\S])`,
  ].join('|'),
  'gu',
);

const SUB: Record<string, string> = {
  '₀': '0',
  '₁': '1',
  '₂': '2',
  '₃': '3',
  '₄': '4',
  '₅': '5',
  '₆': '6',
  '₇': '7',
  '₈': '8',
  '₉': '9',
  ᵢ: 'i',
  ⱼ: 'j',
  ₜ: 't',
  ₙ: 'n',
  ₖ: 'k',
};
const SUP: Record<string, string> = {
  '⁰': '0',
  '¹': '1',
  '²': '2',
  '³': '3',
  '⁴': '4',
  '⁵': '5',
  '⁶': '6',
  '⁷': '7',
  '⁸': '8',
  '⁹': '9',
  '⁻': '−',
  '⁺': '+',
  ˣ: 'x',
  ʸ: 'y',
  ⁿ: 'n',
  ᵀ: 'T',
  ᵏ: 'k',
  ⁱ: 'i',
  ᵘ: 'u',
};

/** Unicode sub/superscript characters as real <sub>/<sup>: the math font has no glyphs for them. */
function scripts(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let run = '';
  let kind: 'sub' | 'sup' | 'base' = 'base';
  const flush = () => {
    if (!run) return;
    if (kind === 'sub') out.push(<sub key={out.length}>{run}</sub>);
    else if (kind === 'sup') out.push(<sup key={out.length}>{run}</sup>);
    else out.push(run);
    run = '';
  };
  for (const ch of text) {
    const next = ch in SUB ? 'sub' : ch in SUP ? 'sup' : 'base';
    if (next !== kind) flush();
    kind = next;
    run += SUB[ch] ?? SUP[ch] ?? ch;
  }
  flush();
  return out;
}

/** Typesets one formula: italic variables, upright functions and numbers, spaced operators. */
function Formula({ src, wrap }: { src: string; wrap: boolean }) {
  const out: ReactNode[] = [];
  let i = 0;
  let space = false; // source whitespace between two terms, e.g. "log softmax"
  let afterOp = true;
  const term = (node: ReactNode) => {
    if (space && !afterOp) out.push('\u202f');
    out.push(node);
    space = afterOp = false;
  };
  for (const m of src.matchAll(TOKEN)) {
    const g = m.groups!;
    const text = m[0];
    if (g.word) {
      const name = text.match(/^[A-Za-z_]+/)![0];
      const rest = text.slice(name.length);
      // A run like "xW" that isn't a known function is a product of variables.
      if (FUNCTIONS.has(name))
        term(
          <span key={i++} className="m-fn">
            {name}
            {scripts(rest)}
          </span>,
        );
      else
        [...name].forEach((ch, k) =>
          term(
            <var key={i++}>
              {ch}
              {k === name.length - 1 ? scripts(rest) : null}
            </var>,
          ),
        );
    } else if (g.ident) term(<var key={i++}>{scripts(text)}</var>);
    else if (g.num)
      term(
        <span key={i++} className="m-num">
          {scripts(text)}
        </span>,
      );
    else if (g.op) {
      out.push(
        // A leading minus (−Σ, (−x) is a sign, not a binary operator: no spacing.
        <span key={i++} className={afterOp && '−-'.includes(text) ? 'm-sign' : 'm-op'}>
          {text === '-' ? '−' : text}
        </span>,
      );
      // A narrow panel may wrap a long inline formula after an operator, never inside a term.
      // Inline formulas may wrap after any operator; displayed ones only between equalities.
      if (wrap || text === '=' || text === '≈') out.push(<wbr key={i++} />);
      space = false;
      afterOp = true;
    } else if (/\s/.test(text)) space = true;
    else {
      out.push(<Fragment key={i++}>{scripts(text)}</Fragment>);
      space = false;
      afterOp = '(['.includes(text);
    }
  }
  return <>{out}</>;
}

/**
 * Renders player-facing copy with formulas. `$…$` marks an inline formula; a line that is
 * exactly `$$…$$` becomes a displayed formula. Other newlines are kept as line breaks.
 */
export function MathText({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <>
      {lines.map((line, n) => {
        const display = line.trim().match(/^\$\$(.+)\$\$$/);
        if (display)
          return (
            <span key={n} className="math-block" role="math">
              <Formula src={display[1]} wrap={false} />
            </span>
          );
        // Punctuation that ends a displayed formula would start the next line alone; drop it.
        if (n > 0 && /^\s*\$\$/.test(lines[n - 1]))
          line = line.replace(/^\s*[。，、；：,.;:]\s*/, '');
        if (!line && n > 0 && /^\s*\$\$/.test(lines[n - 1])) return null;
        const parts = line.split(/\$([^$]+)\$/);
        return (
          <Fragment key={n}>
            {parts.map((part, k) =>
              k % 2 ? (
                <span key={k} className="math" role="math">
                  <Formula src={part} wrap />
                </span>
              ) : (
                part
              ),
            )}
            {n < lines.length - 1 && !lines[n + 1].trim().startsWith('$$') && <br />}
          </Fragment>
        );
      })}
    </>
  );
}
