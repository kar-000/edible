import { useRef, useState } from 'react'

interface Props {
  onSubmit: (file: File, lat?: number, lon?: number) => void
  disabled: boolean
}

async function geocodeZip(zip: string): Promise<{ lat: number; lon: number } | null> {
  const res = await fetch(
    `https://nominatim.openstreetmap.org/search?postalcode=${encodeURIComponent(zip)}&country=US&format=json&limit=1`,
    { headers: { 'User-Agent': 'EdibleApp/1.0 (educational foraging identifier)' } },
  )
  if (!res.ok) throw new Error('Geocoding request failed')
  const data: Array<{ lat: string; lon: string }> = await res.json()
  if (!data.length) return null
  return { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon) }
}

type LocationMode = 'default' | 'zip' | 'manual'

export function ImageUpload({ onSubmit, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)

  // GPS
  const [geoState, setGeoState] = useState<'idle' | 'loading' | 'error'>('idle')
  const [gpsActive, setGpsActive] = useState(false)
  const [gpsLat, setGpsLat] = useState<number | null>(null)
  const [gpsLon, setGpsLon] = useState<number | null>(null)

  // Location input mode
  const [locationMode, setLocationMode] = useState<LocationMode>('default')

  // ZIP
  const [zip, setZip] = useState('')
  const [zipGeocoding, setZipGeocoding] = useState(false)
  const [zipError, setZipError] = useState<string | null>(null)
  const [zipCoords, setZipCoords] = useState<{ lat: number; lon: number } | null>(null)

  // Manual lat/lon
  const [lat, setLat] = useState('')
  const [lon, setLon] = useState('')

  function requestLocation() {
    if (!navigator.geolocation) {
      setGeoState('error')
      return
    }
    setGeoState('loading')
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGpsLat(pos.coords.latitude)
        setGpsLon(pos.coords.longitude)
        setGeoState('idle')
        setGpsActive(true)
      },
      () => setGeoState('error'),
    )
  }

  function clearGps() {
    setGpsLat(null)
    setGpsLon(null)
    setGeoState('idle')
    setGpsActive(false)
  }

  async function handleZipChange(value: string) {
    setZip(value)
    setZipCoords(null)
    setZipError(null)
    if (value.replace(/\D/g, '').length === 5) {
      setZipGeocoding(true)
      try {
        const coords = await geocodeZip(value.trim())
        setZipCoords(coords)
        if (!coords) setZipError('ZIP code not found.')
      } catch {
        setZipError('Could not look up ZIP code. Check your connection.')
      } finally {
        setZipGeocoding(false)
      }
    }
  }

  function openMode(mode: LocationMode) {
    setLocationMode(mode)
    if (mode !== 'zip') { setZip(''); setZipCoords(null); setZipError(null) }
    if (mode !== 'manual') { setLat(''); setLon('') }
  }

  function handleFile(f: File) {
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    if (gpsActive && gpsLat !== null && gpsLon !== null) {
      onSubmit(file, gpsLat, gpsLon)
      return
    }
    if (zipCoords) {
      onSubmit(file, zipCoords.lat, zipCoords.lon)
      return
    }
    const parsedLat = lat ? parseFloat(lat) : undefined
    const parsedLon = lon ? parseFloat(lon) : undefined
    onSubmit(
      file,
      parsedLat !== undefined && !isNaN(parsedLat) ? parsedLat : undefined,
      parsedLon !== undefined && !isNaN(parsedLon) ? parsedLon : undefined,
    )
  }

  const parsedManualLat = lat ? parseFloat(lat) : NaN
  const parsedManualLon = lon ? parseFloat(lon) : NaN
  const hasManualCoords = !isNaN(parsedManualLat) && !isNaN(parsedManualLon)

  const activeLat = gpsActive ? gpsLat : zipCoords ? zipCoords.lat : hasManualCoords ? parsedManualLat : null
  const activeLon = gpsActive ? gpsLon : zipCoords ? zipCoords.lon : hasManualCoords ? parsedManualLon : null
  const hasCoords = activeLat !== null && activeLon !== null

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {preview
          ? <img src={preview} alt="Preview" className="preview-img" />
          : <p>Drop a berry photo here, or click to select</p>
        }
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]) }}
      />

      <div className="gps-section">
        {gpsActive ? (
          <div className="gps-row">
            <button type="button" className="gps-btn gps-btn--clear" onClick={clearGps}>
              × Remove location
            </button>
          </div>
        ) : (
          <>
            <div className="gps-row">
              <button
                type="button"
                className="gps-btn"
                onClick={requestLocation}
                disabled={geoState === 'loading'}
              >
                {geoState === 'loading' ? '…' : geoState === 'error' ? 'Location unavailable' : '📍 Use my location'}
              </button>
              <button
                type="button"
                className="gps-manual-toggle"
                onClick={() => openMode(locationMode === 'zip' ? 'default' : 'zip')}
              >
                {locationMode === 'zip' ? 'Hide ZIP entry' : 'Enter ZIP code'}
              </button>
              <button
                type="button"
                className="gps-manual-toggle"
                onClick={() => openMode(locationMode === 'manual' ? 'default' : 'manual')}
              >
                {locationMode === 'manual' ? 'Hide manual entry' : 'Enter manually'}
              </button>
            </div>

            {locationMode === 'zip' && (
              <div className="gps-zip">
                <div className="gps-zip-row">
                  <input
                    type="text"
                    inputMode="numeric"
                    placeholder="ZIP code (e.g. 78701)"
                    maxLength={10}
                    value={zip}
                    onChange={(e) => handleZipChange(e.target.value)}
                    className="gps-zip-input"
                    aria-label="ZIP code"
                  />
                  {zipGeocoding && <span className="gps-zip-spinner" aria-label="Looking up ZIP">…</span>}
                </div>
                {zipError && <p className="gps-zip-error" role="alert">{zipError}</p>}
                {zipCoords && (
                  <p className="gps-zip-ok">
                    ✓ Located ({zipCoords.lat.toFixed(3)}°, {zipCoords.lon.toFixed(3)}°)
                  </p>
                )}
              </div>
            )}

            {locationMode === 'manual' && (
              <div className="gps-manual">
                <input
                  type="text"
                  inputMode="decimal"
                  placeholder="Latitude"
                  value={lat}
                  onChange={(e) => setLat(e.target.value)}
                />
                <input
                  type="text"
                  inputMode="decimal"
                  placeholder="Longitude"
                  value={lon}
                  onChange={(e) => setLon(e.target.value)}
                />
              </div>
            )}
          </>
        )}

        <p className={`gps-status gps-status--${hasCoords ? 'active' : 'inactive'}`}>
          {hasCoords && activeLat !== null && activeLon !== null
            ? `📍 Using location (${activeLat.toFixed(3)}°, ${activeLon.toFixed(3)}°) — range checking enabled`
            : 'No location — range checking disabled'}
        </p>
      </div>

      <button type="submit" disabled={!file || disabled} className="identify-btn">
        {disabled ? 'Identifying…' : 'Identify'}
      </button>
    </form>
  )
}
