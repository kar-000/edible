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
      </div>

      <button type="submit" disabled={!file || disabled} className="identify-btn">
        {disabled ? 'Identifying…' : 'Identify'}
      </button>
    </form>
  )
}
