import { formatNumber, type Locale } from '../i18n';

/** Compact tensor previews: suppress float noise, retaining at most three significant digits. */
export function formatValue(locale: Locale, value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : '−∞';
  if (Math.abs(value) < 1e-4) return '0';
  const rounded = Number(value.toPrecision(3));
  if (Math.abs(rounded) >= 1e5 || Math.abs(rounded) < 1e-3)
    return (Number.isFinite(rounded) ? rounded.toExponential() : value.toExponential(2))
      .replace(/\.?(0+)(?=e)/, '')
      .replace('e+', 'e');
  return formatNumber(locale, rounded, { maximumSignificantDigits: 3, useGrouping: false });
}
