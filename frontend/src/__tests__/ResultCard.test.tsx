import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { ResultCard } from '../components/ResultCard'
import {
  acceptedEdible,
  acceptedEdibleMidConf,
  acceptedEdibleAtFloor,
  acceptedEdibleBelowFloor,
  acceptedToxic,
  rejectedLowConf,
  rejectedNotPlant,
  rejectedNoFruit,
  acceptedEdibleOutOfRange,
  rejectedOutOfRange,
  acceptedEdibleWithLookalike,
} from '../test/fixtures'

const IMAGE_URL = 'blob:mock-image-url'
const onReset = vi.fn()

function renderCard(result: Parameters<typeof ResultCard>[0]['result']) {
  return render(<ResultCard result={result} imageUrl={IMAGE_URL} onReset={onReset} />)
}

// ─── Safety invariants ────────────────────────────────────────────

describe('SAFETY: rejected results', () => {
  it('shows a danger banner for low_confidence rejection', () => {
    renderCard(rejectedLowConf)
    expect(screen.getAllByText(/do not eat/i).length).toBeGreaterThan(0)
  })

  it('NEVER shows species name when rejected', () => {
    renderCard(rejectedLowConf)
    expect(screen.queryByRole('heading', { level: 2 })).not.toBeInTheDocument()
  })

  it('NEVER shows edibility badge when rejected', () => {
    renderCard(rejectedLowConf)
    expect(screen.queryByText(/edible/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/toxic/i)).not.toBeInTheDocument()
  })

  it('NEVER shows confidence bar when rejected', () => {
    const { container } = renderCard(rejectedLowConf)
    expect(container.querySelector('.confidence-bar-wrap')).not.toBeInTheDocument()
  })

  it('shows detail message for not_a_plant rejection', () => {
    renderCard(rejectedNotPlant)
    expect(screen.getByText(/does not appear to contain a plant/i)).toBeInTheDocument()
  })

  it('shows detail message for no_fruit_visible rejection', () => {
    renderCard(rejectedNoFruit)
    expect(screen.getByText(/no fruit or berries/i)).toBeInTheDocument()
  })
})

describe('SAFETY: toxic accepted results', () => {
  it('shows a danger banner for toxic species', () => {
    renderCard(acceptedToxic)
    expect(screen.getByText(/DO NOT EAT/)).toBeInTheDocument()
  })

  it('still shows the species name for toxic (user should know what they found)', () => {
    renderCard(acceptedToxic)
    expect(screen.getByRole('heading', { level: 2, name: /pokeweed/i })).toBeInTheDocument()
  })

  it('shows TOXIC edibility badge', () => {
    renderCard(acceptedToxic)
    expect(screen.getByText('TOXIC — Do not eat')).toBeInTheDocument()
  })
})

describe('SAFETY: confidence floor', () => {
  it('shows no warning banner at 93% confidence', () => {
    renderCard(acceptedEdible)
    expect(screen.queryByText(/confidence too low/i)).not.toBeInTheDocument()
  })

  it('shows no warning banner at exactly 75% (the floor)', () => {
    renderCard(acceptedEdibleAtFloor)
    expect(screen.queryByText(/confidence too low/i)).not.toBeInTheDocument()
  })

  it('shows warning banner when confidence is below 75%', () => {
    renderCard(acceptedEdibleBelowFloor)
    expect(screen.getByText(/confidence too low/i)).toBeInTheDocument()
  })
})

describe('SAFETY: out-of-range warnings', () => {
  it('shows out-of-range warning for accepted result when is_out_of_range=true', () => {
    renderCard(acceptedEdibleOutOfRange)
    expect(screen.getByText(/not been confirmed in your area/i)).toBeInTheDocument()
  })

  it('does NOT show out-of-range warning for accepted result when is_out_of_range=false', () => {
    renderCard(acceptedEdible)
    expect(screen.queryByText(/not been confirmed in your area/i)).not.toBeInTheDocument()
  })

  it('shows out-of-range context on low_confidence rejection when is_out_of_range=true', () => {
    renderCard(rejectedOutOfRange)
    expect(screen.getByText(/not typically found in your area/i)).toBeInTheDocument()
  })

  it('does NOT show out-of-range context on non-location rejection', () => {
    renderCard(rejectedNotPlant)
    expect(screen.queryByText(/not typically found in your area/i)).not.toBeInTheDocument()
  })
})

// ─── Accepted edible ──────────────────────────────────────────────

describe('accepted edible result', () => {
  it('renders the submitted image', () => {
    renderCard(acceptedEdible)
    expect(screen.getByRole('img', { name: /submitted photo/i })).toHaveAttribute('src', IMAGE_URL)
  })

  it('shows species common name', () => {
    renderCard(acceptedEdible)
    expect(screen.getByRole('heading', { level: 2, name: /elderberry/i })).toBeInTheDocument()
  })

  it('shows scientific name', () => {
    renderCard(acceptedEdible)
    expect(screen.getByText(/sambucus canadensis/i)).toBeInTheDocument()
  })

  it('shows edibility badge', () => {
    renderCard(acceptedEdible)
    expect(screen.getByText(/edible when cooked/i)).toBeInTheDocument()
  })

  it('shows confidence percentage', () => {
    renderCard(acceptedEdible)
    expect(screen.getByText('93% confidence')).toBeInTheDocument()
  })

  it('shows the disclaimer', () => {
    renderCard(acceptedEdible)
    expect(screen.getByText(/educational tool only/i)).toBeInTheDocument()
  })

  it('"Identify another" button calls onReset', async () => {
    const user = userEvent.setup()
    renderCard(acceptedEdible)
    await user.click(screen.getByRole('button', { name: /identify another/i }))
    expect(onReset).toHaveBeenCalledOnce()
  })
})

// ─── Confidence bar semantic color classes ────────────────────────

describe('confidence bar color classes', () => {
  it('applies --high class at 93% confidence', () => {
    const { container } = renderCard(acceptedEdible)
    expect(container.querySelector('.confidence-bar')).toHaveClass('confidence-bar--high')
  })

  it('applies --mid class at 78% confidence', () => {
    const { container } = renderCard(acceptedEdibleMidConf)
    expect(container.querySelector('.confidence-bar')).toHaveClass('confidence-bar--mid')
  })

  it('applies --low class below 75% confidence', () => {
    const { container } = renderCard(acceptedEdibleBelowFloor)
    expect(container.querySelector('.confidence-bar')).toHaveClass('confidence-bar--low')
  })
})

// ─── Look-alike warnings ──────────────────────────────────────────

describe('look-alike warnings', () => {
  it('renders look-alike section when warnings present', () => {
    renderCard(acceptedEdibleWithLookalike)
    expect(screen.getAllByText(/pokeweed/i).length).toBeGreaterThan(0)
  })

  it('does not render look-alike section when no warnings', () => {
    renderCard(acceptedEdible)
    expect(screen.queryByText(/look-alike warnings/i)).not.toBeInTheDocument()
  })
})
