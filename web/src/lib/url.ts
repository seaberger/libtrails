// Base-path-aware URL helper
// In production, BASE_URL is "/libtrails/"; in dev, it's "/".
const base = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

/** Prepend the base path to an internal route. */
export function url(path: string): string {
  return `${base}${path}`;
}
