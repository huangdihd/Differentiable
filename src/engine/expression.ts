/** Standalone expression-to-graph compiler. No evaluation or dynamic code execution. */
export interface Endpoint {
  node: string;
  port: string;
}
export interface ExpressionNode {
  id: string;
  component: string;
  attrs?: Record<string, unknown>;
}
export interface ExpressionEdge {
  from: Endpoint;
  to: Endpoint;
}
export interface ExpressionGraph {
  nodes: ExpressionNode[];
  edges: ExpressionEdge[];
  output: Endpoint;
}

export type BinaryOperator = '+' | '-' | '*' | '/';
/** An operand index (0 or 1), or a component whose inputs are recursively lowered. */
export type Lowering =
  | 0
  | 1
  | {
      component: string;
      inputs: Record<string, Lowering>;
      output?: string;
      attrs?: Record<string, unknown>;
    };
export interface FunctionSignature {
  component: string;
  /** Ordered input port names; their count is the required arity. */
  inputs: readonly string[];
  output?: string;
  attrs?: Record<string, unknown>;
}
export interface ExpressionOptions {
  variables?: Readonly<Record<string, Endpoint>>;
  /** Overrides individual defaults. Set an entry to null to disable that operator. */
  operators?: Partial<Record<BinaryOperator, Lowering | null>>;
  unaryMinus?: Lowering | null;
  functions?: Readonly<Record<string, FunctionSignature>>;
  /** If supplied, every emitted component, including nested lowerings, must exist. */
  availableComponents?: Iterable<string>;
  constant?: { component?: string; output?: string };
  /** Generated IDs skip all variable endpoint node IDs. Default: expr_. */
  idPrefix?: string;
}
export type ExpressionErrorKey =
  | 'unexpected_token'
  | 'unexpected_end'
  | 'invalid_number'
  | 'unknown_variable'
  | 'unknown_function'
  | 'arity_mismatch'
  | 'unsupported_operator'
  | 'missing_component'
  | 'invalid_lowering';
/** Positions are zero-based UTF-16 offsets in the source. Params are localization data. */
export class ExpressionError extends Error {
  readonly operator?: string;
  readonly requiredComponent?: string;
  constructor(
    readonly key: ExpressionErrorKey,
    readonly params: Readonly<Record<string, string | number>>,
    readonly position: number,
  ) {
    super(`${key} at ${position}`);
    this.name = 'ExpressionError';
    this.operator = typeof params.operator === 'string' ? params.operator : undefined;
    this.requiredComponent =
      typeof params.requiredComponent === 'string' ? params.requiredComponent : undefined;
  }
}

const defaults: Record<BinaryOperator, Lowering> = {
  '+': { component: 'add', inputs: { a: 0, b: 1 } },
  '-': { component: 'add', inputs: { a: 0, b: { component: 'neg', inputs: { x: 1 } } } },
  '*': { component: 'mul', inputs: { a: 0, b: 1 } },
  '/': { component: 'mul', inputs: { a: 0, b: { component: 'recip', inputs: { x: 1 } } } },
};
const defaultNeg: Lowering = { component: 'neg', inputs: { x: 0 } };

/** Direct a - b / a / b lowerings onto the player's own sub/div components. */
export const subtractComponent: Lowering = { component: 'sub', inputs: { a: 0, b: 1 } };
export const divideComponent: Lowering = { component: 'div', inputs: { a: 0, b: 1 } };

/**
 * Operator overrides for the editor's expression box: once the player has built (and may use)
 * sub/div, `a - b` and `a / b` insert that single component instead of add+neg / mul+recip.
 */
export function componentOperators(
  available: Iterable<string>,
): NonNullable<ExpressionOptions['operators']> {
  const names = new Set(available);
  return {
    ...(names.has('sub') ? { '-': subtractComponent } : {}),
    ...(names.has('div') ? { '/': divideComponent } : {}),
  };
}
interface Token {
  text: string;
  position: number;
  kind: 'number' | 'name' | 'symbol' | 'end';
}

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let position = 0;
  while (position < source.length) {
    const rest = source.slice(position);
    const whitespace = /^\s+/.exec(rest);
    if (whitespace) {
      position += whitespace[0].length;
      continue;
    }
    const number = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(rest);
    if (number) {
      const text = number[0];
      if (!Number.isFinite(Number(text)) || /^[eE.]/.test(rest.slice(text.length))) {
        throw new ExpressionError('invalid_number', { value: text }, position);
      }
      tokens.push({ text, position, kind: 'number' });
      position += text.length;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z_0-9]*/.exec(rest);
    if (name) {
      tokens.push({ text: name[0], position, kind: 'name' });
      position += name[0].length;
      continue;
    }
    const text = source[position]!;
    if (!'+-*/(),'.includes(text)) {
      throw new ExpressionError('unexpected_token', { token: text }, position);
    }
    tokens.push({ text, position, kind: 'symbol' });
    position++;
  }
  tokens.push({ text: '', position: source.length, kind: 'end' });
  return tokens;
}

/**
 * Compile numbers, variables, parentheses, unary minus, + - * / and function calls.
 * Multiplication/division bind tighter than addition/subtraction; all are left-associative.
 * Defaults emit const(value, shape: []), add(a,b), mul(a,b), neg(x), recip(x),
 * with output port `out`. Subtraction and division lower to add/neg and mul/recip.
 * Variables reference existing endpoints and do not emit nodes. Functions must be
 * explicitly registered. Throws ExpressionError on invalid source/configuration.
 *
 * @example
 * parseExpression('x - 2', { variables: { x: { node: 'input', port: 'out' } } });
 */
export function parseExpression(source: string, options: ExpressionOptions = {}): ExpressionGraph {
  const tokens = tokenize(source);
  let cursor = 0;
  let nextId = 0;
  const graph: ExpressionGraph = { nodes: [], edges: [], output: { node: '', port: '' } };
  const occupied = new Set(Object.values(options.variables ?? {}).map((endpoint) => endpoint.node));
  const available =
    options.availableComponents === undefined ? undefined : new Set(options.availableComponents);
  const peek = () => tokens[cursor]!;
  const fail = (token: Token, expected?: string): never => {
    throw new ExpressionError(
      token.kind === 'end' ? 'unexpected_end' : 'unexpected_token',
      { token: token.text, ...(expected ? { expected } : {}) },
      token.position,
    );
  };
  const expect = (text: string) => {
    if (peek().text !== text) fail(peek(), text);
    cursor++;
  };
  const emit = (
    component: string,
    inputs: Record<string, Endpoint>,
    token: Token,
    output = 'out',
    attrs?: Record<string, unknown>,
    operator?: string,
  ): Endpoint => {
    if (available && !available.has(component)) {
      throw new ExpressionError(
        'missing_component',
        {
          requiredComponent: component,
          ...(operator ? { operator } : {}),
        },
        token.position,
      );
    }
    let id: string;
    do {
      id = `${options.idPrefix ?? 'expr_'}${nextId++}`;
    } while (occupied.has(id));
    occupied.add(id);
    graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, from] of Object.entries(inputs)) {
      graph.edges.push({ from: { ...from }, to: { node: id, port } });
    }
    return { node: id, port: output };
  };
  const lower = (
    rule: Lowering,
    operands: Endpoint[],
    token: Token,
    active = new Set<Lowering>(),
  ): Endpoint => {
    if (typeof rule === 'number') {
      const operand = operands[rule];
      if (!operand)
        throw new ExpressionError(
          'invalid_lowering',
          { operator: token.text, operand: rule },
          token.position,
        );
      return operand;
    }
    if (active.has(rule))
      throw new ExpressionError('invalid_lowering', { operator: token.text }, token.position);
    active.add(rule);
    const inputs: Record<string, Endpoint> = Object.create(null);
    for (const [port, input] of Object.entries(rule.inputs))
      inputs[port] = lower(input, operands, token, active);
    active.delete(rule);
    return emit(rule.component, inputs, token, rule.output, rule.attrs, token.text);
  };
  const primary = (): Endpoint => {
    const token = peek();
    if (token.text === '-') {
      cursor++;
      const rule = options.unaryMinus === undefined ? defaultNeg : options.unaryMinus;
      if (rule === null)
        throw new ExpressionError('unsupported_operator', { operator: '-' }, token.position);
      return lower(rule, [primary()], token);
    }
    if (token.text === '(') {
      cursor++;
      const result = expression(0);
      expect(')');
      return result;
    }
    if (token.kind === 'number') {
      cursor++;
      return emit(options.constant?.component ?? 'const', {}, token, options.constant?.output, {
        value: Number(token.text),
        shape: [],
      });
    }
    if (token.kind === 'name') {
      cursor++;
      if (peek().text === '(') {
        const signature =
          options.functions && Object.hasOwn(options.functions, token.text)
            ? options.functions[token.text]
            : undefined;
        if (!signature)
          throw new ExpressionError('unknown_function', { name: token.text }, token.position);
        cursor++;
        const args: Endpoint[] = [];
        if (peek().text !== ')') {
          args.push(expression(0));
          while (peek().text === ',') {
            cursor++;
            args.push(expression(0));
          }
        }
        expect(')');
        if (args.length !== signature.inputs.length) {
          throw new ExpressionError(
            'arity_mismatch',
            {
              name: token.text,
              expected: signature.inputs.length,
              actual: args.length,
            },
            token.position,
          );
        }
        const inputs = Object.fromEntries(
          signature.inputs.map((port, index) => [port, args[index]!]),
        );
        return emit(signature.component, inputs, token, signature.output, signature.attrs);
      }
      const endpoint =
        options.variables && Object.hasOwn(options.variables, token.text)
          ? options.variables[token.text]
          : undefined;
      if (!endpoint)
        throw new ExpressionError('unknown_variable', { name: token.text }, token.position);
      return { ...endpoint };
    }
    return fail(token);
  };
  const expression = (minimum: number): Endpoint => {
    let left = primary();
    while (true) {
      const token = peek();
      const precedence =
        token.text === '+' || token.text === '-'
          ? 1
          : token.text === '*' || token.text === '/'
            ? 2
            : -1;
      if (precedence < minimum) return left;
      cursor++;
      const operator = token.text as BinaryOperator;
      const override = options.operators?.[operator];
      const rule = override === undefined ? defaults[operator] : override;
      if (rule === null)
        throw new ExpressionError('unsupported_operator', { operator }, token.position);
      const right = expression(precedence + 1);
      // `1 / x` is just recip(x): drop the literal 1 instead of emitting mul(1, recip(x)).
      const literal = graph.nodes.find((n) => n.id === left.node);
      if (
        operator === '/' &&
        (override === undefined || override === divideComponent) &&
        (!available || available.has('recip')) &&
        literal &&
        literal.component === (options.constant?.component ?? 'const') &&
        literal.attrs?.value === 1 &&
        !graph.edges.some((e) => e.from.node === literal.id)
      ) {
        graph.nodes.splice(graph.nodes.indexOf(literal), 1);
        left = emit('recip', { x: right }, token, undefined, undefined, token.text);
        continue;
      }
      left = lower(rule, [left, right], token);
    }
  };
  graph.output = expression(0);
  if (peek().kind !== 'end') fail(peek(), 'end of expression');
  return graph;
}
