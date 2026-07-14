import { useRef, useState } from 'react'

interface Props {
  onSubmit: (file: File, lat?: number, lon?: number) => void
  disabled: boolean
}

export function ImageUpload({ onSubmit, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [lat, setLat] = useState('')
  const [lon, setLon] = useState('')
  const [dragging, setDragging] = useState(false)
  const [geoState, setGeoState] = useState<'idle' | 'loading' | 'error'>('idle')

  function requestLocation() {
    if (!navigator.geolocation) {
      setGeoState('error')
      return
    }
    setGeoState('loading')
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLat(String(pos.coords.latitude))
        setLon(String(pos.coords.longitude))
        setGeoState('idle')
      },
      () => setGeoState('error'),
    )
  }

  function clearLocation() {
    setLat('')
    setLon('')
    setGeoState('idle')
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
    const parsedLat = lat ? parseFloat(lat) : undefined
    const parsedLon = lon ? parseFloat(lon) : undefined
    onSubmit(file, parsedLat, parsedLon)
  }

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

      <div className="gps-row">
        <input
          type="number"
          placeholder="Latitude (optional)"
          value={lat}
          onChange={(e) => setLat(e.target.value)}
          step="any"
        />
        <input
          type="number"
          placeholder="Longitude (optional)"
          value={lon}
          onChange={(e) => setLon(e.target.value)}
          step="any"
        />
        {lat && lon ? (
          <button
            type="button"
            className="gps-btn gps-btn--clear"
            onClick={clearLocation}
            title="Remove location"
          >
            × Remove location
          </button>
        ) : (
          <button
            type="button"
            className="gps-btn"
            onClick={requestLocation}
            disabled={geoState === 'loading'}
            title="Use my current location"
          >
            {geoState === 'loading' ? '…' : geoState === 'error' ? 'Location unavailable' : '📍 Use my location'}
          </button>
        )}
      </div>

      <p className={`gps-status gps-status--${lat && lon ? 'active' : 'inactive'}`}>
        {lat && lon
          ? `📍 Using location (${parseFloat(lat).toFixed(3)}°, ${parseFloat(lon).toFixed(3)}°) — range checking enabled`
          : 'No location — range checking disabled'}
      </p>

      <button type="submit" disabled={!file || disabled} className="identify-btn">
        {disabled ? 'Identifying…' : 'Identify'}
      </button>
    </form>
  )
}
