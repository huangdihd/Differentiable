import en from './locales/en.json';
import zhCN from './locales/zh-CN.json';

/** Add a JSON dictionary here to register a language. Flat keys; plural values use CLDR categories. */
export const locales = { en, 'zh-CN': zhCN };
export type Locale = keyof typeof locales;
export type TranslationParams = Readonly<Record<string, string | number>>;
type Message = string | Partial<Record<Intl.LDMLPluralRule, string>>;
type Dictionary = Record<string, Message>;
/** Native language labels follow the registry without a second registration list. */
export const localeNames = Object.fromEntries(
  Object.keys(locales).map((locale) => [
    locale,
    new Intl.DisplayNames([locale], { type: 'language' }).of(locale) ?? locale,
  ]),
) as Record<Locale, string>;
export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && Object.hasOwn(locales, value);
}
export function matchLocale(languages: readonly string[]): Locale {
  for (const language of languages) {
    const tag = language.toLowerCase();
    const exact = (Object.keys(locales) as Locale[]).find((locale) => locale.toLowerCase() === tag);
    if (exact) return exact;
    const base = (Object.keys(locales) as Locale[]).find(
      (locale) => locale.split('-')[0].toLowerCase() === tag.split('-')[0],
    );
    if (base) return base;
  }
  return 'en';
}
export function browserLocale(): Locale {
  try {
    return matchLocale(navigator.languages ?? [navigator.language]);
  } catch {
    return 'en';
  }
}
export function formatNumber(
  locale: Locale,
  value: number,
  options?: Intl.NumberFormatOptions,
): string {
  return new Intl.NumberFormat(locale, options).format(value);
}
const warned = new Set<string>();
function warn(key: string, locale: Locale) {
  const development = (import.meta as ImportMeta & { env?: { DEV?: boolean } }).env?.DEV;
  const id = `${locale}:${key}`;
  if (development && !warned.has(id)) {
    warned.add(id);
    console.warn(`[i18n] Missing translation: ${id}`);
  }
}
export function translate(locale: Locale, key: string, params: TranslationParams = {}): string {
  const dictionary: Dictionary = locales[locale];
  let messageLocale: Locale = locale;
  let message = Object.hasOwn(dictionary, key) ? dictionary[key] : undefined;
  if (!message) {
    warn(key, locale);
    message = Object.hasOwn(en, key) ? (en as Dictionary)[key] : undefined;
    messageLocale = 'en';
  }
  if (!message) return key;
  const template =
    typeof message === 'string'
      ? message
      : (message[
          new Intl.PluralRules(messageLocale).select(
            typeof params.count === 'number' ? params.count : 0,
          )
        ] ??
        message.other ??
        key);
  return template.replace(/\{(\w+)\}/g, (placeholder, name: string) => {
    if (!Object.hasOwn(params, name)) return placeholder;
    const value = params[name];
    return typeof value === 'number' ? formatNumber(locale, value) : value;
  });
}
export function createTranslator(locale: Locale) {
  return (key: string, params?: TranslationParams) => translate(locale, key, params);
}

/** Raw GraphError/ExpressionError codes map to dictionary keys. Namespaced engine errors pass through. */
export const diagnosticKeys: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(
    Object.keys(en)
      .filter((key) => key.startsWith('errors.'))
      .map((key) => [key.slice(7), key]),
  ),
);
export function diagnosticKey(code: string): string {
  return Object.hasOwn(diagnosticKeys, code)
    ? diagnosticKeys[code]
    : Object.hasOwn(en, code)
      ? code
      : 'errors.unknown';
}
/** Never renders unlocalized backend exception messages. */
export function translateDiagnostic(
  locale: Locale,
  error: {
    key?: string;
    message?: string;
    node?: string;
    port?: string;
    params?: Record<string, unknown>;
  },
): string {
  const params: Record<string, string | number> = {};
  for (const [key, value] of Object.entries(error.params ?? {})) {
    if (key !== 'message' && (typeof value === 'string' || typeof value === 'number'))
      params[key] = value;
    else if (Array.isArray(value) && value.every((d) => typeof d === 'number'))
      params[key] = value.length ? `[${value.join('×')}]` : '[]';
  }
  if (error.node) params.node = error.node;
  if (error.port) params.port = error.port;
  return translate(locale, diagnosticKey(error.key ?? error.message ?? ''), params);
}
