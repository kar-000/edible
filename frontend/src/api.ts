import type { InferenceResult } from './types'

export async function identify(
  file: File,
  lat?: number,
  lon?: number,
): Promise<InferenceResult> {
  const form = new FormData()
  form.append('file', file)
  if (lat !== undefined) form.append('lat', String(lat))
  if (lon !== undefined) form.append('lon', String(lon))

  const res = await fetch('/identify', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json() as Promise<InferenceResult>
}
