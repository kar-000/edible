import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import { LookAlikeCard } from '../components/LookAlikeCard'
import { lookalike } from '../test/fixtures'

describe('LookAlikeCard', () => {
  it('shows the look-alike common name in the header', () => {
    render(<LookAlikeCard warning={lookalike} />)
    expect(screen.getByText(/look-alike: pokeweed/i)).toBeInTheDocument()
  })

  it('shows the severity badge', () => {
    render(<LookAlikeCard warning={lookalike} />)
    expect(screen.getByText('HIGH')).toBeInTheDocument()
  })

  it('shows the warning message by default', () => {
    render(<LookAlikeCard warning={lookalike} />)
    expect(screen.getByText(/highly toxic/i)).toBeInTheDocument()
  })

  it('hides distinguishing features by default', () => {
    render(<LookAlikeCard warning={lookalike} />)
    expect(screen.queryByText(/woody and grey/i)).not.toBeInTheDocument()
  })

  it('hides poison control link by default', () => {
    render(<LookAlikeCard warning={lookalike} />)
    expect(screen.queryByText(/1-800-222-1222/)).not.toBeInTheDocument()
  })

  it('shows features after clicking the toggle', async () => {
    const user = userEvent.setup()
    render(<LookAlikeCard warning={lookalike} />)
    await user.click(screen.getByRole('button', { name: /how to tell them apart/i }))
    expect(screen.getByText(/woody and grey/i)).toBeInTheDocument()
  })

  it('shows poison control link after expanding', async () => {
    const user = userEvent.setup()
    render(<LookAlikeCard warning={lookalike} />)
    await user.click(screen.getByRole('button', { name: /how to tell them apart/i }))
    expect(screen.getByRole('link', { name: /1-800-222-1222/ })).toHaveAttribute('href', 'tel:1-800-222-1222')
  })

  it('collapses features when toggled again', async () => {
    const user = userEvent.setup()
    render(<LookAlikeCard warning={lookalike} />)
    await user.click(screen.getByRole('button', { name: /how to tell them apart/i }))
    await user.click(screen.getByRole('button', { name: /hide details/i }))
    expect(screen.queryByText(/woody and grey/i)).not.toBeInTheDocument()
  })

  it('sets aria-expanded correctly', async () => {
    const user = userEvent.setup()
    render(<LookAlikeCard warning={lookalike} />)
    const toggle = screen.getByRole('button', { name: /how to tell them apart/i })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await user.click(toggle)
    expect(screen.getByRole('button', { name: /hide details/i })).toHaveAttribute('aria-expanded', 'true')
  })

  it('renders all distinguishing features when expanded', async () => {
    const user = userEvent.setup()
    render(<LookAlikeCard warning={lookalike} />)
    await user.click(screen.getByRole('button', { name: /how to tell them apart/i }))
    for (const feature of lookalike.distinguishing_features) {
      expect(screen.getByText(feature)).toBeInTheDocument()
    }
  })
})
