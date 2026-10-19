/**
 * Formatting and arithmetic for MIMIC-IV's timezone-naive clinical timestamps.
 *
 * The backend intentionally serializes hospital wall-clock times without a UTC
 * offset.  Treating them as browser-local Date strings would make their display
 * depend on the viewer's timezone.  These helpers use UTC only as a neutral
 * arithmetic container; they never relabel the clinical time as UTC.
 */
const CLINICAL_TIME = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?$/;

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function clinicalTimeToMs(value: string | null | undefined): number {
  if (!value) return Number.NaN;
  const match = value.trim().replace(/Z$/, "").match(CLINICAL_TIME);
  if (!match) return Number.NaN;
  const [, year, month, day, hour, minute, second = "00"] = match;
  return Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second));
}

export function formatClinicalTimestamp(value: string | null | undefined): string {
  const ms = clinicalTimeToMs(value);
  return Number.isNaN(ms) ? "—" : formatClinicalTimestampFromMs(ms);
}

export function formatClinicalTimestampFromMs(ms: number): string {
  if (Number.isNaN(ms)) return "—";
  const date = new Date(ms);
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
}

export function formatClinicalDate(value: string | null | undefined): string {
  const ms = clinicalTimeToMs(value);
  if (Number.isNaN(ms)) return "—";
  const date = new Date(ms);
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
}

export function toDatetimeLocal(value: string | null | undefined): string {
  const ms = clinicalTimeToMs(value);
  if (Number.isNaN(ms)) return "";
  const date = new Date(ms);
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
}

export function datetimeLocalToClinicalTime(value: string): string {
  const ms = clinicalTimeToMs(value);
  return Number.isNaN(ms) ? value.trim() : formatClinicalTimestampFromMs(ms).replace(" ", "T");
}

export function midpointClinicalTime(first: string, last: string): string | null {
  const firstMs = clinicalTimeToMs(first);
  const lastMs = clinicalTimeToMs(last);
  if (Number.isNaN(firstMs) || Number.isNaN(lastMs) || lastMs <= firstMs) return null;
  return formatClinicalTimestampFromMs((firstMs + lastMs) / 2).replace(" ", "T");
}
