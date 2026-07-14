import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ImageUpload } from '../components/ImageUpload'

const onSubmit = vi.fn()

function renderUpload(disabled = false) {
  return render(<ImageUpload onSubmit={onSubmit} disabled={disabled} />)
}

function makeFile(name = 'berry.jpg') {
  return new File(['content'], name, { type: 'image/jpeg' })
}

beforeEach(() => { onSubmit.mockReset() })

// ─── Initial state ────────────────────────────────────────────────

describe('initial state', () => {
  it('shows the dropzone prompt', () => {
    renderUpload()
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })

  it('identify button is disabled with no file', () => {
    renderUpload()
    expect(screen.getByRole('button', { name: /identify/i })).toBeDisabled()
  })

  it('shows the Use my location button', () => {
    renderUpload()
    expect(screen.getByRole('button', { name: /use my location/i })).toBeInTheDocument()
  })

  it('shows inactive GPS status', () => {
    renderUpload()
    expect(screen.getByText(/no location/i)).toBeInTheDocument()
  })

  it('does not show manual input fields by default', () => {
    renderUpload()
    expect(screen.queryByPlaceholderText(/latitude/i)).not.toBeInTheDocument()
  })
})

// ─── File selection ───────────────────────────────────────────────

describe('file selection', () => {
  it('enables the identify button after selecting a file', async () => {
    const user = userEvent.setup()
    const { container } = renderUpload()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(input, makeFile())
    expect(screen.getByRole('button', { name: /identify/i })).toBeEnabled()
  })

  it('shows image preview after selecting a file', async () => {
    const user = userEvent.setup()
    const { container } = renderUpload()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(input, makeFile())
    expect(screen.getByRole('img', { name: /preview/i })).toBeInTheDocument()
  })
})

// ─── Form submission ──────────────────────────────────────────────

describe('form submission', () => {
  it('calls onSubmit with file and no coordinates when no location set', async () => {
    const user = userEvent.setup()
    const { container } = renderUpload()
    const file = makeFile()
    await user.upload(container.querySelector('input[type="file"]') as HTMLInputElement, file)
    await user.click(screen.getByRole('button', { name: /identify/i }))
    expect(onSubmit).toHaveBeenCalledWith(file, undefined, undefined)
  })

  it('calls onSubmit with coordinates when manually entered', async () => {
    const user = userEvent.setup()
    const { container } = renderUpload()
    const file = makeFile()
    await user.upload(container.querySelector('input[type="file"]') as HTMLInputElement, file)

    await user.click(screen.getByRole('button', { name: /enter manually/i }))
    await user.type(screen.getByPlaceholderText(/latitude/i), '30.267')
    await user.type(screen.getByPlaceholderText(/longitude/i), '-97.743')

    await user.click(screen.getByRole('button', { name: /identify/i }))
    expect(onSubmit).toHaveBeenCalledWith(file, 30.267, -97.743)
  })
})

// ─── GPS ─────────────────────────────────────────────────────────

describe('GPS — Use my location', () => {
  it('populates coordinates on successful geolocation', async () => {
    const user = userEvent.setup()
    vi.mocked(navigator.geolocation.getCurrentPosition).mockImplementationOnce(
      (success) => success({ coords: { latitude: 30.267, longitude: -97.743 } } as GeolocationPosition)
    )
    renderUpload()
    await user.click(screen.getByRole('button', { name: /use my location/i }))
    await waitFor(() => {
      expect(screen.getByText(/using location/i)).toBeInTheDocument()
    })
  })

  it('shows "Remove location" button after location is set', async () => {
    const user = userEvent.setup()
    vi.mocked(navigator.geolocation.getCurrentPosition).mockImplementationOnce(
      (success) => success({ coords: { latitude: 30.267, longitude: -97.743 } } as GeolocationPosition)
    )
    renderUpload()
    await user.click(screen.getByRole('button', { name: /use my location/i }))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /remove location/i })).toBeInTheDocument()
    })
  })

  it('clears location when "Remove location" is clicked', async () => {
    const user = userEvent.setup()
    vi.mocked(navigator.geolocation.getCurrentPosition).mockImplementationOnce(
      (success) => success({ coords: { latitude: 30.267, longitude: -97.743 } } as GeolocationPosition)
    )
    renderUpload()
    await user.click(screen.getByRole('button', { name: /use my location/i }))
    await waitFor(() => screen.getByRole('button', { name: /remove location/i }))
    await user.click(screen.getByRole('button', { name: /remove location/i }))
    expect(screen.getByText(/no location/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /use my location/i })).toBeInTheDocument()
  })

  it('shows error state when geolocation is denied', async () => {
    const user = userEvent.setup()
    vi.mocked(navigator.geolocation.getCurrentPosition).mockImplementationOnce(
      (_success, error) => error!({} as GeolocationPositionError)
    )
    renderUpload()
    await user.click(screen.getByRole('button', { name: /use my location/i }))
    await waitFor(() => {
      expect(screen.getByText(/location unavailable/i)).toBeInTheDocument()
    })
  })
})

// ─── Manual entry toggle ──────────────────────────────────────────

describe('manual entry toggle', () => {
  it('shows manual inputs when "Enter manually" is clicked', async () => {
    const user = userEvent.setup()
    renderUpload()
    await user.click(screen.getByRole('button', { name: /enter manually/i }))
    expect(screen.getByPlaceholderText(/latitude/i)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/longitude/i)).toBeInTheDocument()
  })

  it('hides manual inputs when toggled again', async () => {
    const user = userEvent.setup()
    renderUpload()
    await user.click(screen.getByRole('button', { name: /enter manually/i }))
    await user.click(screen.getByRole('button', { name: /hide manual entry/i }))
    expect(screen.queryByPlaceholderText(/latitude/i)).not.toBeInTheDocument()
  })

  it('hides manual toggle once location is set via button', async () => {
    const user = userEvent.setup()
    vi.mocked(navigator.geolocation.getCurrentPosition).mockImplementationOnce(
      (success) => success({ coords: { latitude: 30.267, longitude: -97.743 } } as GeolocationPosition)
    )
    renderUpload()
    await user.click(screen.getByRole('button', { name: /use my location/i }))
    await waitFor(() => screen.getByRole('button', { name: /remove location/i }))
    expect(screen.queryByRole('button', { name: /enter manually/i })).not.toBeInTheDocument()
  })
})
